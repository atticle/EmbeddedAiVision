## MCU Part
### STM32 설정
pa5 (gpio_output)		led1
pa6 (gpio_output)         led2

----------------------------------
pc7 (tim2_ch4)		red
pb0 (tim1_ch2n)		green
pa7 (tim14_ch1)		blue

Clock Source: Internal Clock
Channel n: PWM Generation CHn
Prescaler (PSC - 1): 64 - 1 (1MHz Timer Clock)
Counter Period (ARR - 1): 1000 - 1 (1kHz PWM frequency)

----------------------------------
pa10 (exti10)		sw1
pb3 (exti3)			sw2

GPIO
  - mode: ...rising/falling edge...

NVIC
  - EXTI line 2 and line 3... : checked
  - EXIT line 4 to 15... : checked
  - USART2... : checked

----------------------------------
pa1 (adc1_in0)		vr
pa2 (adc1_in1)		cds

ADC1
  - SamplingTime Common 1: 160.5 Cycles
  - Number Of Conversion: 2
  - Rank 1
    - Channel: Channel 0
    - Sampling Time: Sampling time common 1
  - Rank 2
    - Channel: Channel 1
    - Sampling Time: Sampling time common 1

DMA settings
  - Add > ADC1
    - Mode: Circular

ADC1
  - Continuous Conversion Mode: Enabled
  - DMA Continuous Requests: Enabled
  - End Of Conversion Selection: End of sequence conversion

### 테스트 코드
- app.c
```c
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"
#include "uart_retarget.h"

extern UART_HandleTypeDef huart2;
extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim14;
extern ADC_HandleTypeDef hadc1;

static volatile bool sw1_clicked = false;
static volatile bool sw2_clicked = false;

void HAL_GPIO_EXTI_Falling_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == GPIO_PIN_10) // SW1 (PA10)
    {
    	if (!sw1_clicked)
    	{
    		sw1_clicked = true;
    		HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
    	}
    }
    else if (GPIO_Pin == GPIO_PIN_3) // SW2 (PB3)
    {
    	if (!sw2_clicked)
    	{
    		sw2_clicked = true;
    		HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_6);
    	}
    }
}

void HAL_GPIO_EXTI_Rising_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == GPIO_PIN_10) // SW1 (PA10)
    {
    	sw1_clicked = false;
    }
    else if (GPIO_Pin == GPIO_PIN_3) // SW2 (PB3)
    {
    	sw2_clicked = false;
    }
}


long clamp(long val, long min, long max)
{
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

long map(long x, long in_min, long in_max, long out_min, long out_max)
{
    if (in_max == in_min) return out_min;
    x = clamp(x, in_min, in_max);
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void rgb(int r, int g, int b)
{
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, map(r, 0, 255, 0, 999));
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_2, map(g, 0, 255, 0, 999));
    __HAL_TIM_SET_COMPARE(&htim14, TIM_CHANNEL_1, map(b, 0, 255, 0, 999));
}


volatile uint16_t adc_values[2];
char buf[128];
int r, g, b;

void setup(void)
{
    UART_Retarget_Init(&huart2);

    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);     // Red (TIM2_CH4)
    HAL_TIMEx_PWMN_Start(&htim1, TIM_CHANNEL_2);  // Green (TIM1_CH2N)
    HAL_TIM_PWM_Start(&htim14, TIM_CHANNEL_1);    // Blue (TIM14_CH1)

    HAL_ADCEx_Calibration_Start(&hadc1);
    HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_values, 2);

    rgb(0, 0, 0);
}


void loop(void)
{
#if 0
    if (fgets(buf, sizeof(buf), stdin) != NULL)
    {
        buf[strcspn(buf, "\r\n")] = '\0';
        if (strlen(buf) != 0)
        {
        	printf(">>> %s\r\n",buf);
        }
    }
#endif

#if 0
    scanf("%d%d%d", &r, &g, &b);
    printf(">>> %d, %d, %d\r\n", r, g, b);
#endif

    printf("VR: %u, CdS: %u\r\n", adc_values[0], adc_values[1]);
    HAL_Delay(100);
}
```

## Raspberry PI 5 Part
### 사진 촬영
1. 웹캠, 50cm ~ 2m , 총 50장
2. 정면(50%), 좌/우 30도 기준(30%), 상/하 15도 필트 (20%)
3. 조명: 어두운/밝은, 안경 착용/미착용
4. 빠른 데이터 수집 팁: 카메라 앞에서 고개를 천천히 돌리는 10 ~ 15초 영상 녹화, opencv 스크립트 작성 5프레임마다 1장씩 파일 저장
   - 단, 배경, 조명 등의 연속성 제거 필요

## 데이터셋(학습/검증)
1. roboflow/ labelImg
2. 팀원 전체 이미지 업로드
3. 라벨링 진행
    - 얼굴 영역에 바운딩 박스(이마 끝 - 턱 끝, 귀 양쪽)
    - 이름 지정
4. 데이터 분할: 학습 80%, 검증 20%
5. 내보내기 YOLOv8
6. 팁: 팀원이 없는 빈 방/빈 책상 화면을 10 ~ 15장 캡처하여, 라벨 파일(.txt)을 빈 내용(0byte)로 둔 채 train/images에 함께 넣어 줌 

### 전이학습 및 ONNX INT 양자화 내보내기
```python
# transfer_learning.py
import os
from ultralytics import YOLO

# 학습 디바이스 선택
import torch
device = 0 if torch.cuda.is_available() else "cpu"
print(f"학습 사용 장치: {device}")

# 전이 학습 및 ONNX 내보내기
model = YOLO("yolov8n.pt")

dataset_yaml = "path/to/dataset/data.yaml"

print("=== 1단계: 전이학습 시작 ===")
model.train(
    data=dataset_yaml,
    epochs=50,
    imgsz=320,
    batch=32,
    cache=True,
    freeze=10,
    workers=4,
    plots=False,
    save_period=-1,
    patience=10,
    device=device
)

print("=== 2단계: 최적화된 FP32 ONNX 생성 ===")
best_pt = os.path.join(model.trainer.save_dir, "weights", "best.pt")
trained_model = YOLO(best_pt)

onnx_fp32_path = trained_model.export(
    format="onnx",
    imgsz=320,
    simplify=True,
    opset=12,
    dynamic=False
)
print(f"FP32 ONNX 경로: {onnx_fp32_path}")

# INT8 양자화
print("=== 3단계: INT8 양자화 수행 ===")
from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_int8_path = onnx_fp32_path.replace(".onnx", "_int8.onnx")

quantize_dynamic(
    model_input=onnx_fp32_path,
    model_output=onnx_int8_path,
    weight_type=QuantType.QInt8,   # 가중치를 int8로 양자화
    optimize_model=True
)
```

### 라즈베리파이5에서 추론
PC에서 생성한 best_int8.onnx를 라즈베리라이5로 복사한 후 진행하며, 파일 구조는 다음과 같다
```text
project/
├── vision_preprocessor_suite2.py  # 아래 수정할 전처리 클래스
├── async_camera_engine.py         # 비동기 카메라 클래스
├── best_int8.onnx                 # 전송받은 양자화 모델
└── yolo_person_inference.py       # 아래 작성할 최종 실행 파일
```

#### vision_preprocessor_suite2.py
```python
# vision_preprocessor_suite2.py
import cv2
import numpy as np

class VisionPreprocessor:
    def __init__(self, target_size=(640, 640), pad_color=(114, 114, 114), norm_mode='yolo', input_dtype=np.float32):
        self.w_dst, self.h_dst = target_size
        self.pad_color = pad_color
        self.input_dtype = input_dtype
        
        # 레터박스용 고정 캔버스 버퍼 (uint8 HWC)
        self.canvas_bgr = np.full((self.h_dst, self.w_dst, 3), pad_color, dtype=np.uint8)
        
        # 정규화 계수 설정
        if norm_mode == 'yolo':
            # [0, 255] -> [0.0, 1.0]
            self.scale = np.float32(1.0 / 255.0)
            self.bias = np.float32(0.0)
        elif norm_mode == 'imagenet':
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            self.scale = (1.0 / (255.0 * std)).astype(np.float32)
            self.bias = (-mean / std).astype(np.float32)
        else:
            self.scale = np.float32(1.0)
            self.bias = np.float32(0.0)

        # 사전 할당 출력 텐서 (NCHW)
        self.out_tensor = np.zeros((1, 3, self.h_dst, self.w_dst), dtype=self.input_dtype)
        self.float_hwc = np.empty((self.h_dst, self.w_dst, 3), dtype=np.float32)

    def letterbox(self, img):
        h_src, w_src = img.shape[:2]

        r = min(self.w_dst / w_src, self.h_dst / h_src)
        w_unpad = int(round(w_src * r))
        h_unpad = int(round(h_src * r))

        pad_w = self.w_dst - w_unpad
        pad_h = self.h_dst - h_unpad
        left = pad_w // 2
        top = pad_h // 2

        # 캔버스 초기화 (기존 버퍼 재사용)
        self.canvas_bgr[:] = self.pad_color
        roi = self.canvas_bgr[top:top + h_unpad, left:left + w_unpad]

        if (w_src, h_src) != (w_unpad, h_unpad):
            cv2.resize(img, (w_unpad, h_unpad), dst=roi, interpolation=cv2.INTER_LINEAR)
        else:
            np.copyto(roi, img)

        params = {
            "ratio": r,
            "pad_left": left,
            "pad_top": top,
            "src_shape": (h_src, w_src)
        }
        return self.canvas_bgr, params

    def transform_to_tensor(self, bgr_img):
        # BGR -> RGB
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.uint8:
            # 양자화 모델이 uint8 입력을 직접 받는 경우
            self.out_tensor[0] = rgb.transpose(2, 0, 1)
        else:
            # 일반적인 YOLO 양자화 모델 (입력 노드는 float32이고 내부 연산만 int8인 형태)
            np.multiply(rgb, self.scale, out=self.float_hwc, dtype=np.float32)
            if isinstance(self.bias, np.ndarray) or self.bias != 0.0:
                self.float_hwc += self.bias
            self.out_tensor[0] = self.float_hwc.transpose(2, 0, 1)

        return self.out_tensor

    def inverse_transform_bboxes(self, boxes, params):
        """
        boxes: np.ndarray shape (N, 4), [x1, y1, x2, y2]
        레터박스 좌표 -> 원본 프레임 좌표 복원 (벡터화 연산)
        """
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.int32)

        r = params["ratio"]
        pad_left = params["pad_left"]
        pad_top = params["pad_top"]
        h_src, w_src = params["src_shape"]

        boxes = boxes.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / r

        # 화면 경계 클리핑
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_src - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_src - 1)

        return boxes.round().astype(np.int32)
```

#### yolo_person_inference.py
```python
import time
import cv2
import numpy as np
import onnxruntime as ort

from vision_preprocessor_suite2 import VisionPreprocessor
from async_camera_engine import HighPerformanceAsyncCamera

# 설정 및 파라미터 정의
MODEL_PATH = "best_int8.onnx"
INPUT_SIZE = (320, 320)       # 30fps 유지를 위한 320x320
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

# data.yaml에 정의했던 클래스 이름 목록
CLASS_NAMES = ["person1", "person2", "person3", "person4"]


def postprocess_yolo(output, conf_thresh, iou_thresh):
    """
    YOLOv8/11 ONNX 출력 형태 (1, 4 + num_classes, num_anchors) 디코딩
    """
    # predictions: (num_anchors, 4 + num_classes)
    predictions = np.squeeze(output, axis=0).T
    
    # [cx, cy, w, h] 추출
    boxes = predictions[:, :4]
    # 클래스 확률 추출
    scores = predictions[:, 4:]
    
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Confidence 필터링
    mask = confidences >= conf_thresh
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
    
    # [cx, cy, w, h] -> [x1, y1, x2, y2] 변환
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = x1 + w
    y2 = y1 + h
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    # cv2.dnn.NMSBoxes용 [x, y, w, h] 정수 리스트 변환
    cv_boxes = [[int(x1[i]), int(y1[i]), int(w[i]), int(h[i])] for i in range(len(boxes))]
    
    indices = cv2.dnn.NMSBoxes(cv_boxes, confidences.tolist(), conf_thresh, iou_thresh)
    
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        return boxes_xyxy[indices], confidences[indices], class_ids[indices]
    
    return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])


def main():
    # ONNX Runtime 세션 초기화 (RPi5 CPU 최적화)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # RPi 5 4코어 모두 활용
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 전처리 파이프라인 및 비동기 카메라 인스턴스 생성
    # 320x320 타겟 사이즈 지정
    preprocessor = VisionPreprocessor(target_size=INPUT_SIZE, norm_mode='yolo', input_dtype=np.float32)
    
    # 640x480 해상도로 카메라 구동 (MJPG 모드로 30~60 FPS 수신)
    camera = HighPerformanceAsyncCamera(src=0, width=640, height=480, codec='MJPG')
    
    fps_eval_time = time.perf_counter()
    frame_counter = 0
    display_fps = 0.0

    print("=== 추론 루프 가동 시작 (종료: 'q') ===")
    try:
        while True:
            # 최신 프레임 획득 (삼중 버퍼 슬롯 스왑)
            success, frame, ts = camera.read_latest(block=True, timeout=0.05)
            if not success or frame is None:
                continue

            # 레터박스 전처리
            canvas_bgr, pad_params = preprocessor.letterbox(frame)

            # 텐서 변환 (NCHW float32)
            input_tensor = preprocessor.transform_to_tensor(canvas_bgr)

            # ONNX Runtime 추론 실행
            outputs = session.run(None, {input_name: input_tensor})

            # NMS 및 디코딩 (레터박스 기준 좌표)
            boxes, confs, cls_ids = postprocess_yolo(outputs[0], CONF_THRESHOLD, IOU_THRESHOLD)

            # 원본 웹캠 해상도(640x480)로 바운딩 박스 역변환
            if len(boxes) > 0:
                orig_boxes = preprocessor.inverse_transform_bboxes(boxes, pad_params)
                
                # 시각화 그리기
                for (x1, y1, x2, y2), conf, cid in zip(orig_boxes, confs, cls_ids):
                    name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"ID_{cid}"
                    label = f"{name} {conf:.2f}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS 측정 및 표시
            frame_counter += 1
            now = time.perf_counter()
            if now - fps_eval_time >= 0.5:
                display_fps = frame_counter / (now - fps_eval_time)
                frame_counter = 0
                fps_eval_time = now

            cv2.putText(frame, f"RPi5 FPS: {display_fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 화면 출력
            cv2.imshow("RPi5 Team Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("=== 자원 정리 및 종료 완료 ===")


if __name__ == "__main__":
    main()
```
