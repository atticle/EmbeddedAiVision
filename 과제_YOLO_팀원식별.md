## 데이터 수집
1. 카메라와의 거리(50cm 근접 ~ 1.5m 원거리)를 골고루 포함해 인원 당 150 ~ 200장 수집
2. 정면(50%), 좌/우 30도 기준(30%), 상/하 15도 필트 (20%)
3. 무표정, 웃는 표정, 말할 때의 입 모양, 안경 착용/미착용, 모자 착용 상태를 섞어서 촬영
4. 실제 추론할 장소의 조명(형광등, 자연광, 역광)과 유사한 환경에서 촬영하여 배경 과적합(Overfitting) 방지
5. 팀 원이 아닌 얼굴 및 빈 공간 사진 100~150장을 데이터셋에 넣되, 라벨 파일(.txt)을 빈 파일로 저장

### 자동 수집 및 오토 라벨링
수천 장의 사진에 일일이 마우스로 바운딩 박스를 그릴 필요 없이, OpenCV 내장 얼굴 검출기(Haar Cascade)로 웹캠 촬영과 YOLO 포맷 라벨링(.txt), 학습/검증(8:2) 분할을 동시에 끝내는 자동화 스크립트를 사용한다.

```python
# dataset_auto_collection.py
import os
import cv2

TARGET_COUNT = 150  # 인당 수집할 목표 이미지 장수
WINDOW_NAME = "Face Data Collector"

# YOLO 데이터셋 디렉토리 구조 자동 생성
for split in ["train", "val"]:
    os.makedirs(f"dataset/images/{split}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split}", exist_ok=True)

# OpenCV 기본 내장 얼굴 검출기 로드
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def save_data_yaml(team_names):
    """수집된 팀원 명단과 인원수에 맞춰 YOLO 학습용 data.yaml 자동 생성"""
    yaml_content = f"""path: {os.path.abspath('dataset')}
train: images/train
val: images/val
names:
"""
    for idx, name in enumerate(team_names):
        yaml_content += f"  {idx}: {name}\n"

    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n[완료] 총 {len(team_names)}명의 팀원 정보가 'data.yaml'에 저장되었습니다: {team_names}")


def collect_member_data(cap, class_id, member_name):
    """단일 팀원의 150장 수집 루프 (2단계 프리뷰 <-> 3단계 검출/저장/취소)"""
    count = 0
    print(f"\n[{member_name} (ID: {class_id})] 카메라 창에서 SPACE를 눌러 얼굴을 캡처하세요.")

    while count < TARGET_COUNT:
        # [2단계] 카메라 프리뷰 화면 출력
        ret, frame = cap.read()
        if not ret or frame is None:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        preview_frame = frame.copy()
        status_text = f"[{member_name} (ID:{class_id})] {count}/{TARGET_COUNT} | Press SPACE to Capture"
        cv2.putText(
            preview_frame,
            status_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, preview_frame)

        key = cv2.waitKey(1) & 0xFF

        # [3단계] SPACE 키를 누르면 현재 프레임에서 얼굴 검출 및 바운딩 박스 출력
        if key == ord(" "):
            h_img, w_img = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            # 얼굴이 검출되지 않은 경우 안내 후 2단계(프리뷰)로 복귀
            if len(faces) == 0:
                warn_frame = frame.copy()
                cv2.putText(
                    warn_frame,
                    "No Face Detected! Try Again.",
                    (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(WINDOW_NAME, warn_frame)
                cv2.waitKey(700)  # 0.7초간 경고 표시 후 프리뷰 복귀
                continue

            # 가장 큰 얼굴 1개 선택 및 바운딩 박스 표시
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces[0]

            review_frame = frame.copy()
            cv2.rectangle(review_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                review_frame,
                f"[{member_name}] {count + 1}/{TARGET_COUNT} -> Press 'S' to Save | 'C' to Cancel",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW_NAME, review_frame)

            # 's'(저장) 또는 'c'(취소) 키 입력 대기
            while True:
                decision_key = cv2.waitKey(0) & 0xFF

                if decision_key in (ord("s"), ord("S")):
                    # 8:2 비율로 train / val 자동 분할 저장
                    split = "val" if (count % 5 == 0) else "train"
                    filename = f"{member_name}_{count:04d}"

                    # YOLO 정규화 좌표 (x_center, y_center, width, height) 계산
                    cx = (x + w / 2.0) / w_img
                    cy = (y + h / 2.0) / h_img
                    nw = w / w_img
                    nh = h / h_img

                    # 박스가 그려지지 않은 깨끗한 원본(frame)과 라벨 저장
                    cv2.imwrite(f"dataset/images/{split}/{filename}.jpg", frame)
                    with open(f"dataset/labels/{split}/{filename}.txt", "w", encoding="utf-8") as f:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                    count += 1
                    print(f"  -> 저장 완료 ({count}/{TARGET_COUNT}): {filename}.jpg")
                    break  # 저장 완료 후 루프 탈출 (150장 미만이면 2단계 프리뷰로 이동)

                elif decision_key in (ord("c"), ord("C")):
                    print("  -> 취소됨 (저장하지 않고 프리뷰로 돌아갑니다).")
                    break  # 취소 후 2단계 프리뷰로 이동


def ask_continue_next_user(cap, completed_member):
    """[4단계] 150장 수집 완료 후 다음 사용자 진행 여부(Y/N)를 묻는 함수"""
    print(f"\n[{completed_member}] {TARGET_COUNT}장 수집 완료!")
    print("다음 사용자 데이터를 수집하시겠습니까? (카메라 창에서 Y: 진행 / N: 종료)")

    ret, frame = cap.read()
    if not ret or frame is None:
        frame = cv2.imread("dataset/images/train/" + os.listdir("dataset/images/train")[0])

    prompt_frame = frame.copy()
    # 가독성을 위한 반투명 어두운 배경 처리
    overlay = prompt_frame.copy()
    cv2.rectangle(overlay, (0, 0), (prompt_frame.shape[1], prompt_frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, prompt_frame, 0.4, 0, prompt_frame)

    cv2.putText(
        prompt_frame,
        f"'{completed_member}' {TARGET_COUNT} Images Completed!",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        prompt_frame,
        "Collect next member? Press [Y] Yes / [N] No",
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    cv2.imshow(WINDOW_NAME, prompt_frame)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("y"), ord("Y")):
            return True
        elif key in (ord("n"), ord("N")):
            return False


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")

    team_names = []
    class_id = 0

    try:
        while True:
            # [1단계] 터미널을 통해 팀원 이름 입력 (창은 잠시 닫아 터미널 포커스 유도)
            cv2.destroyAllWindows()
            while True:
                member_name = input(f"\n[1단계] {class_id}번 팀원의 이름을 입력하세요: ").strip()
                if member_name:
                    break
                print("이름은 비워둘 수 없습니다. 다시 입력해 주세요.")

            team_names.append(member_name)

            # [2단계 ~ 4단계 전반부] 150장이 수집될 때까지 프리뷰 -> SPACE -> S/C 반복
            collect_member_data(cap, class_id, member_name)

            # [4단계 후반부 & 5단계] 150장 수집 완료 후 Y/N 입력에 따라 1단계 반복 또는 종료
            if ask_continue_next_user(cap, member_name):
                class_id += 1
                continue  # 5단계: Y를 누르면 1단계부터 다시 진행
            else:
                print("\n데이터 수집을 종료합니다.")
                break  # N을 누르면 종료

    except KeyboardInterrupt:
        print("\n사용자에 의해 수집이 중단되었습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if team_names:
            save_data_yaml(team_names)


if __name__ == "__main__":
    main()
```

## 전이학습 및 INT8 ONNX 양자화
라즈베리파이 5의 ARM64 CPU(CPUExecutionProvider)에서 ONNX 모델을 INT8로 실행할 때, 동적 양자화(quantize_dynamic)를 사용하면 ConvInteger 연산자 미지원 오류가 발생한다. 따라서 반드시 보정 데이터(Calibration Data)를 활용한 정적 양자화(quantize_static, QDQ 포맷)를 적용해야 하며, YOLO 출력 헤드의 좌표 왜곡을 막기 위해 합성곱(Conv) 및 행렬곱(Gemm, MatMul) 계층만 선택적으로 양자화한다.

```python
# train_and_quantize.py
import glob
import json
import os
import random
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO
import yaml
from vision_preprocessor_suite2 import VisionPreprocessor

IMGSZ = 320
DATA_YAML_PATH = "data.yaml"


def load_and_verify_data_yaml(yaml_path):
    """dataset_auto_collection.py에서 생성된 data.yaml을 로드하고 경로 및 팀원 명단을 검증"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"'{yaml_path}' 파일이 없습니다. 먼저 dataset_auto_collection.py를 실행해 데이터를 수집하세요."
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # 폴더 이동에 대비해 dataset 절대 경로 최신화
    data_cfg["path"] = os.path.abspath("dataset")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    names_dict = data_cfg.get("names", {})
    if isinstance(names_dict, dict):
        team_names = [names_dict[k] for k in sorted(names_dict.keys())]
    else:
        team_names = list(names_dict)

    if not team_names:
        raise ValueError("data.yaml 내부에 등록된 팀원 이름(names)이 없습니다.")

    print(f"[확인 완료] 총 {len(team_names)}명의 팀원 학습을 준비합니다: {team_names}")
    return team_names


class YoloCalibrationDataReader(CalibrationDataReader):
    """모든 팀원의 이미지가 골고루 반영되도록 균등 샘플링하는 INT8 보정 데이터 리더"""

    def __init__(self, image_folders, input_name="images", imgsz=320, max_samples=120):
        all_paths = []
        for folder in image_folders:
            all_paths.extend(glob.glob(os.path.join(folder, "*.jpg")))

        if not all_paths:
            raise RuntimeError("보정(Calibration)에 사용할 이미지가 dataset 폴더에 없습니다.")

        # 특정 팀원에게 편향되지 않도록 고정 시드로 섞은 뒤 샘플링
        random.seed(42)
        random.shuffle(all_paths)
        self.image_paths = all_paths[:max_samples]

        self.input_name = input_name
        self.preprocessor = VisionPreprocessor(
            target_size=(imgsz, imgsz), norm_mode="yolo", input_dtype=np.float32
        )
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            data_list = []
            for img_path in self.image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                lb_img, _ = self.preprocessor.letterbox(img)
                tensor = self.preprocessor.transform_to_tensor(lb_img)
                data_list.append({self.input_name: tensor.copy()})
            self.enum_data = iter(data_list)
        return next(self.enum_data, None)


def embed_team_names_metadata(onnx_path, team_names):
    """라즈베리파이5 추론 코드에서 팀원 명단을 자동 로드할 수 있도록 ONNX 메타데이터에 저장"""
    model = onnx.load(onnx_path)
    meta_dict = {prop.key: prop.value for prop in model.metadata_props}
    meta_dict["team_names_json"] = json.dumps(team_names, ensure_ascii=False)

    del model.metadata_props[:]
    for k, v in meta_dict.items():
        entry = model.metadata_props.add()
        entry.key = str(k)
        entry.value = str(v)

    onnx.save(model, onnx_path)


if __name__ == "__main__":
    # Step 0: 수집된 data.yaml에서 동적 팀원 명단 로드
    team_names = load_and_verify_data_yaml(DATA_YAML_PATH)

    # Step A: YOLOv8n 전이학습 수행 (exist_ok=True로 경로 일관성 유지)
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML_PATH,
        epochs=50,
        imgsz=IMGSZ,
        batch=16,
        patience=15,
        project="face_id",
        name="exp",
        exist_ok=True,
    )

    # Step B: 최고 성능 가중치(best.pt)를 FP32 ONNX로 내보내기
    best_pt = str(model.trainer.best)
    best_model = YOLO(best_pt)
    fp32_onnx = best_model.export(
        format="onnx", imgsz=IMGSZ, simplify=True, opset=17
    )

    # Step C: 양자화 전처리 (Shape Inference & 그래프 최적화)
    prep_onnx = "best_prep.onnx"
    quant_pre_process(fp32_onnx, prep_onnx, skip_symbolic_shape=False)

    # Step D: YOLO Detect Head 좌표 정밀도 보존을 위해 비-연산 노드 양자화 제외
    prep_model = onnx.load(prep_onnx)
    input_name = prep_model.graph.input[0].name
    nodes_to_exclude = [
        n.name
        for n in prep_model.graph.node
        if n.op_type not in {"Conv", "Gemm", "MatMul"}
    ]

    # Step E: ARM64 CPU 최적화 정적 INT8 양자화 (QDQ, U8S8)
    int8_onnx = "best_int8.onnx"
    dr = YoloCalibrationDataReader(
        image_folders=["dataset/images/val", "dataset/images/train"],
        input_name=input_name,
        imgsz=IMGSZ,
        max_samples=120,
    )
    quantize_static(
        model_input=prep_onnx,
        model_output=int8_onnx,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False,
    )

    # Step F: 최종 INT8 ONNX 모델 내부에 팀원 명단 메타데이터 주입 및 임시 파일 정리
    embed_team_names_metadata(int8_onnx, team_names)
    if os.path.exists(prep_onnx):
        os.remove(prep_onnx)

    print(f"\n[양자화 완료] 등록된 팀원 ({len(team_names)}명): {team_names}")
    print(f"라즈베리파이5로 복사할 최종 모델: {os.path.abspath(int8_onnx)}")
```

## 라즈베리파이 5에서 실시간 ONNX 추론
PC에서 생성한 best_int8.onnx를 라즈베리라이5로 복사한 후 진행하며, 파일 구조는 다음과 같다
```text
project/
├── vision_preprocessor_suite2.py  # 전처리 클래스
├── async_camera_engine.py         # 비동기 카메라 클래스
├── best_int8.onnx                 # 양자화 모델
└── yolo_person_inference.py       # 추론 실행 파일
```

신뢰도 임계값(CONF_THRESHOLD = 0.75) 이상인 박스가 검출되면 해당 팀원 이름을 출력하고, 검출되지 않거나 신뢰도가 낮으면(외부인 포함) 즉시 "식별 결과: 없음 (None)"을 출력한다.

### vision_preprocessor_suite2.py
```python
# vision_preprocessor_suite2.py
import cv2
import numpy as np


class VisionPreprocessor:

    def __init__(
        self,
        target_size=(320, 320),
        pad_color=(114, 114, 114),
        norm_mode="yolo",
        input_dtype=np.float32,
    ):
        self.w_dst, self.h_dst = target_size
        self.pad_color = pad_color
        self.input_dtype = input_dtype

        # 레터박스용 고정 캔버스 및 RGB 변환 버퍼 사전 할당 (uint8 HWC)
        self.canvas_bgr = np.full(
            (self.h_dst, self.w_dst, 3), pad_color, dtype=np.uint8
        )
        self.rgb_hwc = np.empty((self.h_dst, self.w_dst, 3), dtype=np.uint8)
        self._last_src_shape = None
        self._cached_params = None

        # 정규화 계수 설정
        if norm_mode == "yolo":
            self.scale = np.float32(1.0 / 255.0)
            self.bias = np.float32(0.0)
        elif norm_mode == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
                1, 1, 3
            )
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(
                1, 1, 3
            )
            self.scale = (1.0 / (255.0 * std)).astype(np.float32)
            self.bias = (-mean / std).astype(np.float32)
        else:
            self.scale = np.float32(1.0)
            self.bias = np.float32(0.0)

        # 사전 할당 출력 텐서 (NCHW)
        self.out_tensor = np.zeros(
            (1, 3, self.h_dst, self.w_dst), dtype=self.input_dtype
        )
        self.float_hwc = np.empty((self.h_dst, self.w_dst, 3), dtype=np.float32)

    def letterbox(self, img):
        h_src, w_src = img.shape[:2]

        # 입력 프레임 해상도가 변경된 경우에만 패딩 좌표 계산 및 캔버스 초기화 수행
        if self._last_src_shape != (h_src, w_src):
            r = min(self.w_dst / w_src, self.h_dst / h_src)
            w_unpad = int(round(w_src * r))
            h_unpad = int(round(h_src * r))

            pad_w = self.w_dst - w_unpad
            pad_h = self.h_dst - h_unpad
            left = pad_w // 2
            top = pad_h // 2

            self.canvas_bgr[:] = self.pad_color
            self._last_src_shape = (h_src, w_src)
            self._cached_params = {
                "ratio": r,
                "pad_left": left,
                "pad_top": top,
                "w_unpad": w_unpad,
                "h_unpad": h_unpad,
                "src_shape": (h_src, w_src),
            }

        p = self._cached_params
        top, left = p["pad_top"], p["pad_left"]
        w_unpad, h_unpad = p["w_unpad"], p["h_unpad"]

        roi = self.canvas_bgr[top : top + h_unpad, left : left + w_unpad]
        if (w_src, h_src) != (w_unpad, h_unpad):
            cv2.resize(
                img, (w_unpad, h_unpad), dst=roi, interpolation=cv2.INTER_LINEAR
            )
        else:
            np.copyto(roi, img)

        return self.canvas_bgr, p

    def transform_to_tensor(self, bgr_img):
        # 사전 할당된 버퍼(self.rgb_hwc)에 BGR -> RGB 변환
        cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB, dst=self.rgb_hwc)

        if self.input_dtype == np.uint8:
            self.out_tensor[0] = self.rgb_hwc.transpose(2, 0, 1)
        else:
            np.multiply(
                self.rgb_hwc, self.scale, out=self.float_hwc, dtype=np.float32
            )
            if isinstance(self.bias, np.ndarray) or self.bias != 0.0:
                self.float_hwc += self.bias
            self.out_tensor[0] = self.float_hwc.transpose(2, 0, 1)

        return self.out_tensor

    def inverse_transform_bboxes(self, boxes, params):
        """boxes: np.ndarray shape (N, 4), [x1, y1, x2, y2]

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

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_src - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_src - 1)

        return boxes.round().astype(np.int32)
```

### async_camera_engine2.py
```python
# async_camera_engine.py
import cv2
import threading
import time

class HighPerformanceAsyncCamera:
    """
    - 삼중 버퍼 슬롯 교환으로 단일 소비 중 프레임 덮어쓰기 방지
    - 프레임 데이터 복사 없이 인덱스와 상태를 락으로 동기화
    - V4L2 캡처 버퍼 크기를 1로 요청 (백엔드 지원 여부에 따름)
    """
    def __init__(self, src=0, width=640, height=480, fps=30, codec='MJPG'):
        self.width = width
        self.height = height

        # V4L2 백엔드 명시
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 장치(/dev/video{src})를 열 수 없습니다.")

        # 카메라 하드웨어 매개변수 설정
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 초기 프레임 확인
        ret, init_frame = self.cap.read()
        if not ret or init_frame is None:
            self.cap.release()
            raise RuntimeError("카메라에서 첫 프레임을 읽어오지 못했습니다.")

        self.running = True
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        # 생산자, 공유, 소비자 역할을 위한 세 슬롯 구성
        # 0: Producer Write Buffer, 1: Shared Latest Buffer, 2: Consumer Read Buffer
        self.pool = [init_frame.copy(), init_frame.copy(), init_frame.copy()]
        self.producer_idx = 0
        self.shared_idx = 1
        self.consumer_idx = 2

        self.new_frame_ready = False
        self.frame_id = 0
        self.last_consumed_frame_id = 0
        self.latest_timestamp = 0.0

        # 통계 지표
        self.produced_frames = 0
        self.consumed_frames = 0
        self.dropped_frames = 0
        self.capture_fps = 0.0

        # 백그라운드 캡처 스레드 가동
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _capture_worker(self):
        fps_count = 0
        fps_timer = time.perf_counter()

        while self.running:
            # 카메라에서 프레임을 읽어 반환받는다.
            ret, frame = self.cap.read()
            capture_time = time.perf_counter()

            if not ret or frame is None:
                time.sleep(0.002)
                continue

            # 반환된 프레임 참조를 생산자 슬롯에 연결
            self.pool[self.producer_idx] = frame

            # 임계 구역에서는 프레임 데이터를 복사하지 않고 슬롯과 상태를 갱신
            with self.condition:
                # 생산자 슬롯과 공유 슬롯의 인덱스를 맞바꾼다.
                self.producer_idx, self.shared_idx = self.shared_idx, self.producer_idx
                self.frame_id += 1
                self.produced_frames += 1
                self.latest_timestamp = capture_time
                self.new_frame_ready = True

                # 대기 중인 소비자에게 새 프레임 도착을 알림
                self.condition.notify()

            # FPS 계측
            fps_count += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                with self.lock:
                    self.capture_fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

    def read_latest(self, block=True, timeout=None):
        """
        소비자가 최신 프레임 참조와 호스트 캡처 시각을 획득하는 함수.
        프레임 참조는 다음 read_latest 호출 전까지 사용하고, 그 이후까지 보관하려면 복사한다.
        
        Returns:
            (success: bool, frame: np.ndarray, timestamp: float)
        """
        with self.condition:
            if block and not self.new_frame_ready:
                if not self.condition.wait(timeout=timeout):
                    return False, None, 0.0

            if not self.new_frame_ready:
                return False, None, 0.0

            # 소비자 슬롯과 공유 슬롯을 교환해 최신 프레임 참조를 획득
            self.consumer_idx, self.shared_idx = self.shared_idx, self.consumer_idx

            # 건너뛴 프레임 통계 계산
            skipped = self.frame_id - self.last_consumed_frame_id - 1
            if skipped > 0:
                self.dropped_frames += skipped
            self.last_consumed_frame_id = self.frame_id
            self.consumed_frames += 1
            self.new_frame_ready = False

            ts = self.latest_timestamp
            current_frame = self.pool[self.consumer_idx]

        return True, current_frame, ts

    def get_stats(self):
        with self.lock:
            return {
                "capture_fps": self.capture_fps,
                "latest_frame_id": self.frame_id,
                "consumed_frame_id": self.last_consumed_frame_id,
                "produced": self.produced_frames,
                "consumed": self.consumed_frames,
                "dropped": self.dropped_frames
            }

    def release(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
            
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.cap.isOpened():
            self.cap.release()
```

### yolo_person_inference.py
```python
# yolo_person_inference.py
import ast
from collections import deque
from datetime import datetime
import json
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from async_camera_engine2 import HighPerformanceAsyncCamera
from vision_preprocessor_suite2 import VisionPreprocessor


MODEL_PATH = "best_int8.onnx"
CONF_THRESHOLD = 0.75  # 개별 프레임 신뢰도 임계값 (미만일 경우 외부인/'없음' 판정)
IOU_THRESHOLD = 0.45

# 스무딩(Smoothing) 파라미터 (30FPS 기준 5프레임 = 약 0.16초)
WINDOW_SIZE = 5  # 최근 5프레임 결과 보관
MIN_HITS = 3  # 5프레임 중 3프레임 이상 검출 시 팀원 확정
EMA_ALPHA = 0.6  # 바운딩 박스 좌표 떨림 보정 계수

# 쿨다운(Cooldown) 및 로그 설정
COOLDOWN_SEC = 15.0  # 동일 팀원 재실행 방지 시간 (초)
LOG_FILE_PATH = "team_access_log.csv"
EVENT_BANNER_SEC = 2.0  # 이벤트 발생 시 화면 알림 배너 유지 시간 (초)


def load_team_names(session, fallback_yaml="data.yaml"):
    """ONNX 모델 메타데이터 또는 data.yaml에서 팀원 명단과 인원수를 동적으로 로드"""
    meta = session.get_modelmeta().custom_metadata_map

    # train_and_quantize.py에서 주입한 JSON 메타데이터
    if "team_names_json" in meta:
        return json.loads(meta["team_names_json"])

    # Ultralytics 기본 ONNX export 메타데이터 ('names')
    if "names" in meta:
        names_dict = ast.literal_eval(meta["names"])
        if isinstance(names_dict, dict):
            return [names_dict[int(k)] for k in sorted(names_dict.keys(), key=int)]
        return list(names_dict)

    # 동일 폴더의 data.yaml 파일 참조
    if os.path.exists(fallback_yaml):
        import yaml

        with open(fallback_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names_dict = cfg.get("names", {})
        if isinstance(names_dict, dict):
            return [names_dict[int(k)] for k in sorted(names_dict.keys(), key=int)]
        return list(names_dict)

    raise RuntimeError(
        "모델 메타데이터나 data.yaml에서 팀원 명단을 찾을 수 없습니다. "
        "train_and_quantize.py로 생성된 모델인지 확인하세요."
    )


class TemporalSmoother:
    """최근 N프레임 다수결 투표(Majority Voting) 및 바운딩 박스 EMA 보정기"""

    def __init__(self, window_size=5, min_hits=3, ema_alpha=0.6):
        self.window_size = window_size
        self.min_hits = min_hits
        self.ema_alpha = ema_alpha
        self.history = deque(maxlen=window_size)
        self.smoothed_boxes = {}
        self.last_scores = {}

    def update(self, frame_detections):
        self.history.append(frame_detections)

        # 현재 프레임 검출 클래스의 EMA 바운딩 박스 및 점수 갱신
        for cls_id, (box, score) in frame_detections.items():
            box_f = box.astype(np.float32)
            if cls_id in self.smoothed_boxes:
                self.smoothed_boxes[cls_id] = (
                    self.ema_alpha * box_f
                    + (1.0 - self.ema_alpha) * self.smoothed_boxes[cls_id]
                )
            else:
                self.smoothed_boxes[cls_id] = box_f
            self.last_scores[cls_id] = score

        # 최근 window_size 프레임 동안 각 클래스별 등장 횟수 집계
        hit_counts = {}
        for past_det in self.history:
            for cls_id in past_det.keys():
                hit_counts[cls_id] = hit_counts.get(cls_id, 0) + 1

        # 윈도우에서 완전히 사라진 클래스 캐시 정리
        active_ids = set(hit_counts.keys())
        for cached_id in list(self.smoothed_boxes.keys()):
            if cached_id not in active_ids:
                del self.smoothed_boxes[cached_id]
                del self.last_scores[cached_id]

        # MIN_HITS 이상 등장한 클래스만 최종 팀원으로 확정
        confirmed = []
        for cls_id, hits in hit_counts.items():
            if hits >= self.min_hits:
                confirmed.append({
                    "cls_id": cls_id,
                    "box": self.smoothed_boxes[cls_id].round().astype(np.int32),
                    "score": self.last_scores[cls_id],
                    "hits": hits,
                    "is_held": cls_id not in frame_detections,
                })

        return confirmed


class EventCooldownManager:
    """팀원별 독립 쿨다운 타이머 관리 및 CSV 이벤트 로거"""

    def __init__(self, team_names, cooldown_sec=15.0, log_path="team_access_log.csv"):
        self.team_names = team_names
        self.cooldown_sec = cooldown_sec
        self.log_path = log_path
        self.last_trigger_time = {}  # {cls_id: perf_counter_timestamp}

        # 최근 발생한 이벤트 화면 배너 표시용 상태
        self.last_event_msg = ""
        self.last_event_ts = -9999.0

        # CSV 파일 초기화 (파일이 없거나 비어 있으면 헤더 작성)
        file_exists = os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0
        self.log_file = open(self.log_path, mode="a", encoding="utf-8", buffering=1)
        if not file_exists:
            self.log_file.write("timestamp,member_id,member_name,confidence,hits\n")
            self.log_file.flush()

    def on_member_verified(self, cls_id, name, score, hits, timestamp_str):
        """[단발성 이벤트 훅] 팀원이 확정되었을 때 쿨다운 주기당 딱 1회만 호출됩니다.

        GPIO 도어락 제어, 부저/TTS 출력, 네트워크 API 호출 등을 이곳에 작성하세요.
        """
        print(
            f"[EVENT TRIGGERED] {timestamp_str} | ID: {cls_id} ({name}) "
            f"| Conf: {score:.2f} | Hits: {hits}"
        )

    def process(self, confirmed_results, now):
        """확정된 팀원 리스트를 받아 쿨다운 여부를 판별하고 신규 이벤트를 트리거합니다."""
        for item in confirmed_results:
            cls_id = item["cls_id"]
            name = (
                self.team_names[cls_id]
                if cls_id < len(self.team_names)
                else f"ID_{cls_id}"
            )
            score = item["score"]
            hits = item["hits"]

            last_ts = self.last_trigger_time.get(cls_id, -9999.0)
            elapsed = now - last_ts

            if elapsed >= self.cooldown_sec:
                # 쿨다운 만료 또는 첫 인식 -> 이벤트 발생 및 로그 기록
                self.last_trigger_time[cls_id] = now
                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # CSV 기록
                self.log_file.write(
                    f"{ts_str},{cls_id},{name},{score:.4f},{hits}\n"
                )
                self.log_file.flush()

                # 외부 연동 이벤트 훅 실행
                self.on_member_verified(cls_id, name, score, hits, ts_str)

                # 화면 알림 배너 갱신
                self.last_event_msg = f"VERIFIED: {name} ({ts_str.split(' ')[1]})"
                self.last_event_ts = now

                item["triggered_now"] = True
                item["cooldown_rem"] = 0.0
            else:
                # 쿨다운 진행 중 -> 중복 실행 차단 및 남은 시간 계산
                item["triggered_now"] = False
                item["cooldown_rem"] = self.cooldown_sec - elapsed

        return confirmed_results

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일('{MODEL_PATH}')을 찾을 수 없습니다.")

    # CPU 최적화 ONNX 세션 초기화
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # 모델에서 입력 텐서 정보 및 팀원 명단 동적 추출
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_dtype = np.uint8 if "uint8" in input_info.type else np.float32

    # 입력 shape (1, 3, H, W)에서 모델 해상도 자동 추출 (기본값 320x320)
    in_shape = input_info.shape
    if (
        len(in_shape) == 4
        and isinstance(in_shape[2], int)
        and isinstance(in_shape[3], int)
    ):
        target_size = (in_shape[3], in_shape[2])  # (width, height)
    else:
        target_size = (320, 320)

    team_names = load_team_names(session)
    print(f"[모델 로드 완료] 입력 크기: {target_size} | 등록된 팀원({len(team_names)}명): {team_names}")

    # 전처리기, 비동기 카메라, 5프레임 스무더, 쿨다운 매니저 초기화
    preprocessor = VisionPreprocessor(
        target_size=target_size, norm_mode="yolo", input_dtype=input_dtype
    )

    # async_camera_engine2의 fps 인자 지원 여부 자동 호환
    try:
        camera = HighPerformanceAsyncCamera(
            src=0, width=640, height=480, fps=30, codec="MJPG"
        )
    except TypeError:
        camera = HighPerformanceAsyncCamera(
            src=0, width=640, height=480, codec="MJPG"
        )

    smoother = TemporalSmoother(
        window_size=WINDOW_SIZE, min_hits=MIN_HITS, ema_alpha=EMA_ALPHA
    )
    event_manager = EventCooldownManager(
        team_names=team_names,
        cooldown_sec=COOLDOWN_SEC,
        log_path=LOG_FILE_PATH,
    )

    print(f"실시간 팀원 식별 시작 (로그 파일: {LOG_FILE_PATH}, 종료: 'q')...")
    prev_loop_time = time.perf_counter()

    try:
        while True:
            # 비동기 카메라에서 최신 프레임 획득
            success, frame, cap_ts = camera.read_latest(block=True, timeout=1.0)
            if not success or frame is None:
                continue

            # 고성능 레터박스 및 텐서 변환
            letterboxed_bgr, lb_params = preprocessor.letterbox(frame)
            input_tensor = preprocessor.transform_to_tensor(letterboxed_bgr)

            # ONNX Runtime INT8 추론
            outputs = session.run(None, {input_name: input_tensor})[0]
            preds = np.squeeze(outputs).T

            # 현재 프레임 후처리 (Confidence 필터링 -> NMS -> 레터박스 역변환)
            class_scores = preds[:, 4:]
            max_scores = np.max(class_scores, axis=1)
            valid_mask = max_scores >= CONF_THRESHOLD

            current_frame_dets = {}

            if np.any(valid_mask):
                v_preds = preds[valid_mask]
                v_scores = max_scores[valid_mask]
                v_class_ids = np.argmax(v_preds[:, 4:], axis=1)

                cx, cy, w, h = (
                    v_preds[:, 0],
                    v_preds[:, 1],
                    v_preds[:, 2],
                    v_preds[:, 3],
                )
                x1 = cx - w * 0.5
                y1 = cy - h * 0.5

                nms_boxes = np.column_stack((x1, y1, w, h)).tolist()
                indices = cv2.dnn.NMSBoxes(
                    nms_boxes, v_scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD
                )

                if len(indices) > 0:
                    idx = indices.flatten()
                    x2 = x1[idx] + w[idx]
                    y2 = y1[idx] + h[idx]
                    boxes_xyxy_lb = np.column_stack(
                        (x1[idx], y1[idx], x2, y2)
                    )
                    orig_boxes = preprocessor.inverse_transform_bboxes(
                        boxes_xyxy_lb, lb_params
                    )

                    for i, box in enumerate(orig_boxes):
                        cls_id = int(v_class_ids[idx[i]])
                        score = float(v_scores[idx[i]])
                        if (
                            cls_id not in current_frame_dets
                            or score > current_frame_dets[cls_id][1]
                        ):
                            current_frame_dets[cls_id] = (box, score)

            # 5프레임 스무딩 업데이트 및 쿨다운/이벤트 처리
            now = time.perf_counter()
            confirmed_results = smoother.update(current_frame_dets)
            confirmed_results = event_manager.process(confirmed_results, now)

            confirmed_names = []

            for item in confirmed_results:
                cls_id = item["cls_id"]
                bx1, by1, bx2, by2 = item["box"]
                score = item["score"]
                cd_rem = item["cooldown_rem"]
                name = (
                    team_names[cls_id]
                    if cls_id < len(team_names)
                    else f"ID_{cls_id}"
                )
                confirmed_names.append(name)

                # 상태별 라벨 및 색상 구분 (이벤트 발생 / 쿨다운 대기 / 스무딩 홀드)
                if item["is_held"]:
                    box_color = (0, 180, 255)  # 주황색: 순간 가려짐 유지 상태
                else:
                    box_color = (0, 255, 0)  # 초록색: 실시간 검출 상태

                status_tag = (
                    "EVENT!" if item["triggered_now"] else f"CD:{cd_rem:.0f}s"
                )
                label_text = f"{name} ({score:.2f} | {status_tag})"

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
                cv2.putText(
                    frame,
                    label_text,
                    (bx1, max(20, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2,
                )

            # 상단 최종 결과 판정 (팀원 이름 vs None)
            if confirmed_names:
                unique_names = ", ".join(sorted(set(confirmed_names)))
                result_str = f"Team: {unique_names}"
                text_color = (0, 255, 0)
            else:
                result_str = "Result: None"
                text_color = (0, 0, 255)

            # FPS 및 최근 이벤트 배너 표시
            infer_fps = 1.0 / max(now - prev_loop_time, 1e-6)
            prev_loop_time = now
            latency_ms = (now - cap_ts) * 1000.0
            cam_stats = camera.get_stats()

            cv2.putText(
                frame,
                result_str,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                text_color,
                2,
            )
            cv2.putText(
                frame,
                f"Cam: {cam_stats['capture_fps']:.1f} FPS | Infer: {infer_fps:.1f} FPS | Latency: {latency_ms:.1f}ms",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            # 최근 EVENT_BANNER_SEC(2초) 이내에 기록된 이벤트가 있으면 화면 하단에 배너 출력
            if (now - event_manager.last_event_ts) <= EVENT_BANNER_SEC:
                cv2.rectangle(frame, (15, 425), (625, 465), (0, 120, 0), -1)
                cv2.putText(
                    frame,
                    f"[LOGGED] {event_manager.last_event_msg}",
                    (25, 452),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("RPi5 Async YOLO Face ID (Event & Cooldown)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        event_manager.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```
