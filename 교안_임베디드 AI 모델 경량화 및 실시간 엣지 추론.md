# 교안: 임베디드 AI 모델 경량화 및 실시간 엣지 추론

> **빠른 학습판**  
> 목표: Raspberry Pi에서 카메라 입력 → ONNX 추론을 실행하고, 가지치기·INT8 전후를 같은 조건에서 비교한다.  
> 실습은 작은 기능을 따로 만드는 대신, 각 단계의 최종 산출물을 만든다.

## 1. 핵심 개념

| 단계 | 반드시 기억할 개념 | 완성할 결과 |
|---|---|---|
| 1. 장비 | CPU·메모리·카메라·온도와 냉각이 성능을 제한한다. | 작동하는 카메라와 기준 장비 기록 |
| 2. 영상 입력 | 평균 FPS보다 프레임 나이와 지연이 중요하다. 느린 소비자는 오래된 프레임을 버리고 최신 프레임을 처리한다. | 최신 프레임 카메라 입력과 모델 전처리 |
| 3. ONNX | ONNX는 프레임워크와 추론 런타임 사이의 모델 전달 형식이다. 변환 후 그래프 검사와 출력 비교가 필요하다. | `mobilenet_v2_opt.onnx`와 수치 비교 |
| 4. 추론 | 세션은 한 번 만들고 재사용한다. `session.run()` 시간과 전체 카메라 처리 시간은 다르다. | 실시간 분류 및 지연 통계 |
| 5. 가지치기 | 가중치를 0으로 만들어도 연산은 줄지 않을 수 있다. 실제 채널을 줄여야 밀집 연산 크기가 감소한다. | 희소 모델과 0.5배 폭 축소 모델 비교 |
| 6. 양자화 | FP32를 INT8로 바꾸면 표현 오차와 클리핑이 생긴다. INT8이 항상 더 빠른 것은 아니다. | 보정된 QDQ 모델과 FP32 비교 |

<img src="images/embedded_ai_tradeoff.svg" alt="클라우드 추론과 온디바이스 추론의 흐름 및 주요 제약" style="display:block;width:100%;max-width:1000px;height:auto">

### 측정 원칙

- **세 가지 검증을 섞지 않는다:** 파일 실행 가능성, 같은 입력의 출력 일치, 라벨이 있는 자료의 정확도는 서로 다른 검사다.
- **같은 조건으로 비교한다:** 입력, 전처리, 실행 제공자, 스레드 수, 냉각 상태를 고정하고 모델만 바꾼다.
- **지표를 나누어 기록한다:** 모델 호출 P50/P95/P99, 처리 FPS, 프레임 나이, CPU/RSS/온도를 함께 본다.
- **결과를 일반화하지 않는다:** 한 장비의 FPS, 전력, 온도는 다른 장비의 성능 보장이 아니다.

## 2. 실습 준비와 장비 확인

### 기준 환경

- Raspberry Pi 5 (8GB), 액티브 쿨러, USB UVC 카메라
- tOS-Lite AI의 Python·OpenCV·PyTorch·ONNX·ONNX Runtime 환경
- 기준 모델: torchvision 사전 학습 MobileNetV2, 입력 `[1, 3, 224, 224]`
- 기준 런타임: `CPUExecutionProvider` (전용 NPU를 사용하지 않음)

카메라와 시스템 상태를 먼저 확인한다.

```sh
uname -m
free -h
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

`/dev/video0`은 장비에 따라 달라질 수 있다. 프리뷰에서 실제 영상을 확인하고 해상도·픽셀 형식·FPS를 기록한다. 유휴 상태와 부하 상태에서 CPU·메모리·온도를 확인하며, 장시간 측정에는 냉각 팬을 켠다.

처음 장비를 준비할 때는 tOS-Lite AI 이미지를 microSD에 기록하고, 냉각 팬과 전원을 연결해 부팅한다. 이더넷과 SSH로 원격 접속한 뒤 작업 디렉터리에서 Python·OpenCV·PyTorch·ONNX Runtime을 확인한다. 온도 센서 경로는 운영체제마다 다를 수 있으므로, 값을 읽지 못하면 `N/A`로 숨기지 말고 장비의 센서 경로와 권한을 확인한다. CPU 부하 실습은 팬을 분리하지 않고 온도가 빠르게 오르면 즉시 중단한다.

## 3. 교재 전체를 이해하기 위한 핵심 개념

### 3.1 장비에서 추론까지: 시스템을 먼저 그린다

<img src="images/camera_to_inference.svg" alt="카메라 센서부터 USB/UVC와 V4L2, OpenCV, 전처리, 신경망을 거쳐 결과를 표시하는 흐름" style="display:block;width:100%;max-width:1000px;height:auto">

클라우드 추론은 프레임을 네트워크로 보내므로 서버 자원을 쓸 수 있지만 통신 지연·대역폭·개인정보·네트워크 단절을 고려해야 한다. 온디바이스 추론은 입력을 장치 안에서 처리해 네트워크 의존을 줄이는 대신 장치의 연산·메모리·전력·냉각 한계에 맞춰야 한다.

실습 기준인 Raspberry Pi 5는 4개 Cortex-A76 CPU 코어와 8GB 공유 메모리를 사용하며, 전용 NPU가 있다고 가정하지 않는다. CPU 캐시는 반복 데이터를 메모리에서 다시 읽는 비용을 줄이고, 메모리 대역폭은 모델 가중치와 중간 텐서를 얼마나 빨리 공급할 수 있는지에 영향을 준다. 연속 부하로 칩 온도가 오르면 CPU 클록이 낮아질 수 있으므로, 비교 전후에 온도와 냉각 조건을 기록한다.

ARM NEON은 128비트 벡터 연산 기능이다. 한 벡터에 FP32 값 4개 또는 INT8 값 16개가 들어갈 수 있지만, 이것은 벡터의 원소 수이지 전체 모델이 각각 4배·16배 빨라진다는 뜻은 아니다. 실제 속도는 커널 구현, 메모리 접근, 런타임과 연산 병목으로 측정한다.

USB 카메라는 UVC 장치로 인식되고, V4L2 커널 드라이버와 버퍼를 거쳐 `/dev/video*`로 노출된다. OpenCV `VideoCapture.read()`는 프레임을 사용자 공간의 배열로 가져온다. YUYV 같은 비압축 형식은 픽셀당 바이트가 커서 USB 대역폭을 많이 쓰고, MJPEG는 전송량이 줄 수 있지만 디코딩 비용이 생긴다. 지원 모드 목록과 실제 캡처 FPS를 함께 확인한다.

### 3.2 실시간 영상: FPS와 프레임 나이는 다르다

- **카메라 주기:** `T_cam = 1 / FPS_cam`. 30 FPS면 새 프레임 간격은 약 33.3ms다.
- **처리 시간:** 프레임 읽기·전처리·추론·후처리를 합친 시간이 카메라 주기보다 길면 대기열에 프레임이 쌓일 수 있다.
- **대기 지연 근사:** `D_queue ≈ N_buffer × T_cam`. 30 FPS에서 오래된 프레임 4장이 대기하면 큐만으로 약 133ms가 더해질 수 있다.
- **FPS:** 일정 시간에 처리한 프레임 수. **프레임 나이:** 캡처 이후 표시까지 프레임이 얼마나 오래되었는지. 높은 FPS여도 오래된 프레임이면 화면 반응은 느리다.
- **P50/P95/P99:** 측정값의 중앙값과 꼬리 지연을 나타내는 백분위수. 평균만 보면 드물게 발생하는 긴 지연이 가려질 수 있다.

원 교재의 삼중 버퍼는 생산자(캡처), 공유 최신 프레임, 소비자(추론) 슬롯의 역할을 짧은 동기화로 교환한다. 소비 중인 슬롯을 덮어쓰지 않으며, 큰 이미지 복사 대신 슬롯 역할을 바꾸는 구조다. 아래 실습 코드는 원리를 빠르게 익히도록 `Queue(maxsize=1)`로 단순화했다. 중간 프레임을 버릴 수 있다는 점은 같지만, 메모리 소유권과 스왑 구현까지 삼중 버퍼와 동일한 코드는 아니다.

<img src="images/triple_buffer_slots.svg" alt="캡처·공유·소비 슬롯이 역할을 교환하는 삼중 버퍼 흐름" style="display:block;width:100%;max-width:1000px;height:auto">

Python의 GIL은 Python 바이트코드 병렬 실행을 제한하지만, OpenCV의 카메라 입출력은 C++에서 실행되며 대기 중 GIL을 해제할 수 있다. 실제로 캡처와 추론이 겹쳐지는 정도는 OpenCV 빌드와 장비에서 측정한다.

### 3.3 전처리: 픽셀 배열을 모델 입력 계약에 맞춘다

카메라 영상은 보통 BGR·HWC·`uint8`이고, 이 실습의 MobileNetV2는 RGB·NCHW·`float32` 입력을 받는다. 레터박스는 원본 종횡비를 유지하며 크기를 맞추고 남는 영역을 패딩한다.

원본 크기를 `W_s × H_s`, 모델 입력 크기를 `W_t × H_t`라 하면:

$$r = \min(W_t/W_s,\;H_t/H_s), \qquad W_r=\operatorname{round}(rW_s), \quad H_r=\operatorname{round}(rH_s)$$

좌우·상하 패딩은 `(W_t-W_r)`와 `(H_t-H_r)`를 나누어 배치한다. 물체 상자 좌표를 원본으로 되돌릴 때는 `x_src=(x_model-pad_left)/r`, `y_src=(y_model-pad_top)/r`처럼 패딩을 빼고 배율로 나눈다. 분류만 할 때는 상자 복원이 필요 없지만, 검출·ROI 처리에서는 배율과 패딩을 반드시 보관한다.

MobileNetV2 기준 입력 변환은 BGR→RGB → 레터박스 → `[0,1]` 스케일 → ImageNet 채널 평균 `[0.485, 0.456, 0.406]` 및 표준편차 `[0.229, 0.224, 0.225]`로 정규화 → HWC에서 CHW로 전치 → 배치 축 추가다. 최종 형상은 `[1,3,224,224]`이며, 모델에 넘기기 전에 연속 메모리 배열로 만든다. 다른 모델은 입력 크기와 정규화 값이 다를 수 있으므로 모델 명세를 따른다.

<img src="images/letterbox_preprocessing.svg" alt="영상 레터박스 전처리와 NCHW 텐서 변환" style="display:block;width:100%;max-width:1000px;height:auto">

### 3.4 신경망과 ONNX: 변환 전에 출력의 의미를 이해한다

- **모델:** 계층을 연결한 계산 구조와 학습된 파라미터(가중치·편향)의 묶음. 학습은 파라미터를 조정하고, 추론은 고정된 파라미터로 새 입력을 계산한다.
- **합성곱(Conv):** 입력의 작은 영역에 필터를 적용해 특징 맵을 만든다. ReLU 같은 활성화 함수는 비선형성을 더하고, BatchNorm은 추론 시 학습 때 저장한 통계를 사용한다. Linear 계층은 특징을 클래스 점수로 바꾼다.
- **로짓과 확률:** ImageNet 분류 모델의 출력 `[1,1000]`은 클래스별 로짓이다. Softmax는 로짓을 합이 1인 점수 분포로 바꾸지만, 그 값이 실제 확률처럼 보정되었다고 보장하지 않는다. 정확도는 라벨이 있는 검증 자료로 따로 계산한다.
- **연산량:** MAC은 곱셈-누적 횟수이고, FLOPs는 세는 기준에 따라 대략 MAC의 2배로 볼 수 있다. 같은 MACs라도 실제 지연 시간은 메모리·캐시·커널·스레드 설정에 따라 달라진다.
- **MobileNet:** 표준 합성곱은 대략 `H×W×K²×C_in×C_out` MACs를 쓴다. 깊이별 분리 합성곱은 `H×W×(K²×C_in + C_in×C_out)`로 공간 필터와 채널 혼합을 나누어 계산을 줄인다. MobileNetV2는 확장 1×1 → depthwise 합성곱 → 선형 projection을 사용하는 inverted residual 블록을 쓴다.

ONNX 파일에는 입력·출력 텐서, 연산 노드와 가중치가 담긴다. `opset`은 연산자 규격 버전이다. `onnx.checker`는 그래프의 형식 오류를 검사하고, 상수 접기(constant folding)는 변환 시점에 미리 계산할 수 있는 부분을 단순화한다. 그래프 검사를 통과해도 실행 결과가 원본과 같다는 뜻은 아니므로 같은 입력의 수치 비교를 한다.

### 3.5 ONNX Runtime과 결과 판정

ONNX Runtime은 `InferenceSession`을 만들 때 모델을 읽고 실행 계획을 준비한다. 따라서 세션을 프레임마다 새로 만들지 않고 한 번 만들어 재사용한다. `CPUExecutionProvider`는 CPU 실행 경로다. `SessionOptions`에서 `intra_op_num_threads`와 그래프 최적화 수준을 바꾸어 비교하되, 설정은 세션 생성 때 적용되므로 설정 변경 후 세션을 다시 만들어야 한다.

ONNX Runtime의 CPU 커널(예: MLAS)은 지원되는 경우 NEON을 활용할 수 있다. `session.run()`만 잰 시간은 모델 호출 시간이다. 카메라 대기·전처리·화면 표시를 포함하는 종단 간 시간과 처리 FPS는 별도로 기록한다. 현재 실습 코드의 지연 표기는 전처리와 추론 시간을 따로 보여 준다.

실시간 분류에서는 여러 프레임의 점수를 짧은 창으로 평균해 흔들림을 줄일 수 있다. Top-1 점수가 낮거나 Top-1과 Top-2 차이가 작거나 결과 프레임이 오래되면 새 라벨 표시를 보류한다. 확률·격차·최대 프레임 나이 기준은 검증 자료와 사용 목적에 맞춰 정하며, 모든 모델에 공통인 임계값은 없다.

### 3.6 경량화의 두 방식: 구조와 숫자 표현

**가지치기(Pruning)**는 중요도가 낮다고 판단한 가중치를 제거한다. 비구조적 방식은 개별 가중치를 0으로 만든다.

$$\text{희소율} = \frac{\text{0인 가중치 수}}{\text{전체 가중치 수}} \times 100\%$$

희소율이 늘어도 텐서 형상과 밀집 행렬 계산량은 그대로일 수 있다. 희소 연산 커널이 실제로 0을 건너뛸 때만 속도 개선을 기대할 수 있다. 구조적 가지치기는 필터·채널을 제거해 텐서 형상 자체를 줄인다. 입력·출력 채널이 연결된 계층을 함께 맞춰야 하며, MobileNet depthwise 계층은 `groups=channels` 관계, residual 연결은 더해지는 텐서의 채널 수를 보존해야 한다.

폭 배율 축소는 여러 계층의 채널 수를 일관된 규칙으로 줄이는 구조 변경이다. 이 실습의 0.5배 모델은 가중치를 새 구조에 맞게 잘라 복사한 것이며 재학습 모델이 아니다. 파라미터·MACs 감소가 실제 지연 시간이나 정확도 변화와 같은 비율이라고 가정하지 않는다.

**양자화(Quantization)**는 FP32 실수를 더 작은 정수 표현으로 매핑한다. signed INT8은 -128~127 범위이고, 대칭 양자화 예시에서는 -127~127을 쓴다.

$$q=\operatorname{clip}(\operatorname{round}(r/S)+Z,\;q_{\min},q_{\max}), \qquad r'=S(q-Z)$$

`S`는 scale, `Z`는 zero point다. FP32 가중치는 원소당 4바이트, INT8은 1바이트이므로 가중치 데이터는 작아질 수 있지만, 그래프·scale·zero point 등 부가 정보 때문에 ONNX 파일이 정확히 1/4 크기가 된다고 보장할 수 없다. 동적 양자화는 실행 중 범위를 계산하고, 정적 양자화(PTQ)는 보정 자료로 범위를 미리 계산한다. QDQ의 `QuantizeLinear`·`DequantizeLinear` 노드는 그래프의 변환 위치를 표시하며, QDQ가 있다고 해서 CPU가 모든 연산을 정수 커널로 실행하는 것은 아니다.

보정 자료는 배포 장면을 대표해야 한다. 보정 자료로 변환을 준비하고, 정확도는 별도의 라벨 검증 자료로 평가한다. 모델 크기, 출력 오차, 정확도, 지연 시간은 각각 다른 측정값이다.

<img src="images/neon_simd_accel.svg" alt="128비트 NEON 벡터에 FP32 4개 또는 INT8 16개가 들어가는 구조" style="display:block;width:100%;max-width:1000px;height:auto">

## 4. MobileNetV2를 ONNX로 변환하고 검증

`export_validate.py`로 FP32 기준 모델을 만든다. 처음 실행할 때 사전 학습 가중치 다운로드가 필요할 수 있다.

```python
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import models
import time


model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
).eval()
sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
model_path = Path("mobilenet_v2_opt.onnx")
sample_np = sample.numpy()

torch.onnx.export(
    model,
    sample,
    str(model_path),
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True,
    dynamo=False,
)

onnx.checker.check_model(str(model_path))
options = ort.SessionOptions()
options.intra_op_num_threads = 4
session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
with torch.inference_mode():
    expected = model(sample).cpu().numpy()
actual = session.run(["logits"], {"input": sample_np})[0]

max_error = float(np.max(np.abs(expected - actual)))
print(f"ONNX valid: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
print(f"Maximum absolute output error: {max_error:.6g}")
np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)


def benchmark(run):
    for _ in range(5):
        run()
    times = []
    for _ in range(50):
        started = time.perf_counter()
        run()
        times.append((time.perf_counter() - started) * 1000)
    return np.percentile(times, [50, 95, 99])


torch.set_num_threads(4)
with torch.inference_mode():
    torch_latency = benchmark(lambda: model(sample))
    ort_latency = benchmark(lambda: session.run(["logits"], {"input": sample_np}))
print("PyTorch P50/P95/P99 ms:", torch_latency.round(2))
print("ONNX Runtime P50/P95/P99 ms:", ort_latency.round(2))
```

블록을 `export_validate.py`로 저장하고 실행한다: `python export_validate.py`.

오차 기준은 시작점일 뿐, 모델·런타임에 맞춰 정한다. 검사가 실패하면 임계값만 넓히기 전에 입력 이름·형상·자료형과 변환 그래프를 먼저 확인한다.

<img src="images/onnx_runtime_flow.svg" alt="세션을 한 번 초기화한 뒤 입력 텐서를 반복 추론하는 ONNX Runtime 흐름" style="display:block;width:100%;max-width:1000px;height:auto">

## 5. 최신 프레임 실시간 추론

다음 코드를 `quick_edge_infer.py`로 저장한다. 최대 1개의 미처리 프레임만 보관해 소비자가 늦으면 오래된 프레임을 새 프레임으로 교체한다. 전처리와 ONNX 세션도 한 파일 안에 두어 바로 실행할 수 있다.

```python
import argparse
from collections import deque
from queue import Empty, Full, Queue
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort


SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROB_HISTORY = 5
MIN_TOP1_PROB = 0.30
MIN_TOP1_MARGIN = 0.10
MAX_FRAME_AGE_MS = 200.0  # 예시 기준: 제품 조건에 맞춰 검증할 것


def preprocess(bgr):
    height, width = bgr.shape[:2]
    ratio = min(SIZE / width, SIZE / height)
    new_width = round(width * ratio)
    new_height = round(height * ratio)
    resized = cv2.resize(bgr, (new_width, new_height))

    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    left = (SIZE - new_width) // 2
    top = (SIZE - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def capture_loop(cap, frames, stop, errors):
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            if not stop.is_set():
                errors.append("카메라 프레임을 읽지 못했습니다.")
                stop.set()
            return

        packet = (time.perf_counter(), frame)
        try:
            frames.put_nowait(packet)
        except Full:
            try:
                frames.get_nowait()
            except Empty:
                pass
            frames.put_nowait(packet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_v2_opt.onnx")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        args.model,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {args.camera}")

    frames = Queue(maxsize=1)
    stop = threading.Event()
    errors = []
    worker = threading.Thread(
        target=capture_loop,
        args=(cap, frames, stop, errors),
        daemon=True,
    )
    worker.start()

    infer_ms = deque(maxlen=300)
    prob_history = deque(maxlen=PROB_HISTORY)
    processed = 0
    report_started = time.perf_counter()
    try:
        while not stop.is_set():
            try:
                captured_at, frame = frames.get(timeout=0.2)
            except Empty:
                continue

            preprocess_started = time.perf_counter()
            tensor = preprocess(frame)
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

            inference_started = time.perf_counter()
            logits = session.run(None, {input_name: tensor})[0][0]
            elapsed_ms = (time.perf_counter() - inference_started) * 1000
            age_ms = (time.perf_counter() - captured_at) * 1000
            infer_ms.append(elapsed_ms)

            scores = logits - np.max(logits)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
            prob_history.append(probabilities)
            smoothed = np.mean(prob_history, axis=0)
            top2 = np.argsort(smoothed)[-2:][::-1]
            class_id, second_id = int(top2[0]), int(top2[1])
            confidence = float(smoothed[class_id])
            margin = confidence - float(smoothed[second_id])
            accepted = (
                confidence >= MIN_TOP1_PROB
                and margin >= MIN_TOP1_MARGIN
                and age_ms <= MAX_FRAME_AGE_MS
            )
            label = (
                f"class {class_id} ({confidence:.2f})"
                if accepted else "uncertain or stale"
            )

            cv2.putText(
                frame,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"prep {preprocess_ms:.1f} ms | infer {elapsed_ms:.1f} ms | age {age_ms:.1f} ms",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Edge inference (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            processed += 1
            now = time.perf_counter()
            if now - report_started >= 5 and infer_ms:
                p50, p95, p99 = np.percentile(infer_ms, [50, 95, 99])
                fps = processed / (now - report_started)
                print(
                    f"processed FPS={fps:.1f}, "
                    f"session.run P50/P95/P99="
                    f"{p50:.1f}/{p95:.1f}/{p99:.1f} ms"
                )
                processed = 0
                report_started = now
    finally:
        stop.set()
        cap.release()
        worker.join(timeout=2)
        cv2.destroyAllWindows()

    if errors:
        raise RuntimeError(errors[0])
    if worker.is_alive():
        raise RuntimeError("카메라 캡처 스레드가 종료되지 않았습니다.")


if __name__ == "__main__":
    main()
```

실행 예:

```sh
python quick_edge_infer.py --model mobilenet_v2_opt.onnx --threads 4
```

표시되는 `class N`은 ImageNet 클래스 번호다. 화면의 softmax 값은 보정된 신뢰도라고 보장되지 않는다. 이 예제의 프레임 나이는 `cap.read()` 반환 시점부터 계산하므로 센서 노출부터의 전체 지연은 아니다. 제품에서는 라벨 매핑, 오류 복구, 장비별 온도·RSS 수집을 추가한다. FPS가 높아도 프레임 나이가 크면 반응이 늦을 수 있다.

<img src="images/video_buffer_latency.svg" alt="오래된 프레임을 쌓는 방식과 최신 프레임을 우선하는 방식 비교" style="display:block;width:100%;max-width:1000px;height:auto">

### 같은 추론기로 속도 비교

같은 실행 파일을 모델만 바꾸어 실행한다. 입력·카메라 모드·스레드 수를 고정하고 각 모델을 충분히 워밍업한 뒤 30초 이상 측정한다.

```sh
python quick_edge_infer.py --model mobilenet_v2_opt.onnx --threads 4
python quick_edge_infer.py --model mobilenet_v2_w05.onnx --threads 4
python quick_edge_infer.py --model mobilenet_v2_int8.onnx --threads 4
```

카메라의 장면은 실행마다 달라지므로, 예측 결과 비교에는 보정용이 아닌 별도의 정지 영상 하나를 사용한다. 아래 코드를 `compare_models.py`로 저장하면 한 입력으로 최적화 모델들의 호출 지연과 출력 차이를 비교할 수 있다.

```python
import time

import cv2
import numpy as np
import onnxruntime as ort

from quick_edge_infer import preprocess


frame = cv2.imread("comparison.jpg")
if frame is None:
    raise FileNotFoundError("comparison.jpg를 읽을 수 없습니다.")
sample = preprocess(frame)


def benchmark(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    feed = {input_name: sample}
    for _ in range(5):
        session.run(None, feed)

    times = []
    for _ in range(50):
        started = time.perf_counter()
        output = session.run(None, feed)[0]
        times.append((time.perf_counter() - started) * 1000)
    return np.percentile(times, [50, 95, 99]), output


models_to_compare = [
    "mobilenet_v2_opt.onnx",
    "mobilenet_v2_sparse.onnx",
    "mobilenet_v2_w05.onnx",
    "mobilenet_v2_int8.onnx",
]
reference = None
for path in models_to_compare:
    percentiles, output = benchmark(path)
    if reference is None:
        reference = output
    max_diff = float(np.max(np.abs(reference - output)))
    print(
        f"{path}: P50/P95/P99={percentiles.round(2)} ms, "
        f"max output difference={max_diff:.6g}"
    )
```

이 코드는 한 장면의 **출력 차이**를 보여 줄 뿐 정확도를 평가하지 않는다. 서비스 처리량과 프레임 나이는 별도로 `quick_edge_infer.py`에서 측정한다.

## 6. 모델 구조 경량화

### 비구조적 가지치기: 희소성은 속도 향상을 보장하지 않는다

아래 코드를 `prune_sparse.py`로 저장해 작은 가중치를 0으로 바꾸고 희소 모델을 만든다. 일반 밀집 CPU 커널은 0도 계산할 수 있으므로 파일 크기와 실행 시간을 측정한다.

```python
import onnx
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
).eval()
parameters = [
    (module, "weight")
    for module in model.modules()
    if isinstance(module, (nn.Conv2d, nn.Linear))
]
prune.global_unstructured(
    parameters,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
for module, name in parameters:
    prune.remove(module, name)

sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
torch.onnx.export(
    model,
    sample,
    "mobilenet_v2_sparse.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True,
    dynamo=False,
)
onnx.checker.check_model("mobilenet_v2_sparse.onnx")
```

### 폭 배율 0.5배 모델 만들기

모든 계층의 채널 수를 독립적으로 줄이지 말고 MobileNetV2가 제공하는 0.5배 구조를 사용한다. 가중치 앞부분을 대상 텐서 모양에 맞게 복사한 뒤 ONNX로 내보낸다.

```python
import torch
import onnx
from torchvision import models


base = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
).eval()
small = models.mobilenet_v2(width_mult=0.5, weights=None).eval()

base_state = base.state_dict()
small_state = small.state_dict()
for key, target in small_state.items():
    source = base_state[key]
    if source.shape != target.shape:
        source = source[tuple(slice(0, size) for size in target.shape)]
    small_state[key] = source.clone()
small.load_state_dict(small_state)

print("1.0x parameters:", sum(p.numel() for p in base.parameters()))
print("0.5x parameters:", sum(p.numel() for p in small.parameters()))

sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
torch.onnx.export(
    small,
    sample,
    "mobilenet_v2_w05.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True,
    dynamo=False,
)
onnx.checker.check_model("mobilenet_v2_w05.onnx")
```

블록을 `build_w05.py`로 저장하고 `python build_w05.py`로 실행한다.

이 가중치 복사는 구조와 실행을 확인하기 위한 빠른 실습이다. 0.5배 모델은 재학습·미세 조정한 모델이 아니므로 정확도를 주장하지 않는다. 구조 변경 뒤에는 출력 규격 검사, ONNX 실행, 라벨이 있는 품질 평가를 각각 수행한다.

<img src="images/pruning_concepts.png" alt="밀집 모델, 비구조적 가지치기, 구조적 가지치기의 차이" style="display:block;width:100%;max-width:1000px;height:auto">

## 7. 대표 입력으로 INT8 정적 양자화

`calibration/`에 실제 사용 장면을 대표하는 JPG 프레임 20장을 준비한다. 어두움·밝음, 거리, 배경이 한 종류에 치우치지 않게 고르고, 정확도 검증용 자료와는 분리한다. 아래 코드를 `build_int8.py`로 저장한다. `quick_edge_infer.py`의 `preprocess()`를 재사용하므로 두 파일을 같은 폴더에 둔다.

```python
from pathlib import Path

import cv2
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from quick_edge_infer import preprocess


model_in = Path("mobilenet_v2_opt.onnx")
model_out = Path("mobilenet_v2_int8.onnx")
if not model_in.is_file():
    raise FileNotFoundError(f"기준 모델이 없습니다: {model_in}")

input_name = ort.InferenceSession(
    str(model_in),
    providers=["CPUExecutionProvider"],
).get_inputs()[0].name

samples = []
for path in sorted(Path("calibration").glob("*.jpg"))[:20]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {path}")
    samples.append({input_name: preprocess(frame)})
if len(samples) < 20:
    raise RuntimeError("calibration/ 폴더에 읽을 수 있는 JPG 20장이 필요합니다.")


class ImageReader(CalibrationDataReader):
    def __init__(self, items):
        self.items = iter(items)

    def get_next(self):
        return next(self.items, None)


quantize_static(
    model_input=str(model_in),
    model_output=str(model_out),
    calibration_data_reader=ImageReader(samples),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
check = ort.InferenceSession(
    str(model_out),
    providers=["CPUExecutionProvider"],
)
check.run(None, samples[0])
print(f"Created {model_out} ({model_out.stat().st_size / 1e6:.1f} MB)")
```

양자화 수식은 `q = clip(round(r / scale) + zero_point, q_min, q_max)`이다. 정적 보정은 입력 프레임에서 활성화 범위를 추정한다. 보정 자료가 실제 장면을 대표하지 않으면 변환이 성공해도 정확도는 보장되지 않는다.

<img src="images/quantization_flow.svg" alt="FP32 실수를 INT8로 양자화하고 근사값으로 복원하는 과정" style="display:block;width:100%;max-width:1000px;height:auto">

## 8. 결과를 정확도까지 검증하기

동일한 입력 한 장의 출력 차이는 디버깅에 유용하지만 정확도 평가를 대신하지 않는다. 변환·가지치기·폭 축소·양자화 전후를 비교하려면 라벨이 있는 **별도 검증 자료**를 같은 전처리로 실행한다. 다음처럼 `validation/labels.csv`에 이미지 파일명과 ImageNet 정답 인덱스를 기록하고, 이 자료는 INT8 보정에 사용하지 않는다.

```csv
image,class_id
sample_001.jpg,281
sample_002.jpg,207
```

아래 코드는 모델의 Top-1 정확도를 계산한다. 표본 수와 장면 구성을 결과와 함께 기록한다.

```python
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from quick_edge_infer import preprocess


def top1_accuracy(model_path, labels_path="validation/labels.csv"):
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    labels_file = Path(labels_path)
    correct = 0
    total = 0
    with labels_file.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_path = labels_file.parent / row["image"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"검증 이미지를 읽지 못했습니다: {image_path}")
            logits = session.run(None, {input_name: preprocess(frame)})[0][0]
            correct += int(np.argmax(logits) == int(row["class_id"]))
            total += 1
    if total == 0:
        raise RuntimeError("검증 자료가 비어 있습니다.")
    return correct / total, total


for model_path in (
    "mobilenet_v2_opt.onnx",
    "mobilenet_v2_sparse.onnx",
    "mobilenet_v2_w05.onnx",
    "mobilenet_v2_int8.onnx",
):
    accuracy, count = top1_accuracy(model_path)
    print(f"{model_path}: Top-1={accuracy:.3f}, images={count}")
```

## 9. 전체 학습 흐름과 이해 확인

원 교재 본문 1~6장의 흐름과 이 교안의 대응은 다음과 같다. 이 절은 부록을 포함하지 않는다.

| 원 교재 본문 | 이 교안에서 이해할 내용 | 확인할 산출물 |
|---|---|---|
| 1장: 엣지 장비·카메라·온도 | 2절, 3.1절: 장비 제약, V4L2/UVC, 형식·온도·부하 | 카메라 입력과 유휴/부하 상태 기록 |
| 2장: 실시간 영상 처리 | 3.2~3.3절, 5절: 프레임 나이, 큐, 전처리·좌표 계약 | 최신 프레임 추론 화면과 지연 지표 |
| 3장: 신경망·ONNX | 3.4~3.5절, 4절: MobileNetV2, 그래프, 실행 세션, 출력 동등성 | `mobilenet_v2_opt.onnx`와 오차 |
| 4장: 추론 결과·응답성 | 3.5절, 5절: Top-1/Top-2, 시간 창 평균, 오래된 결과 보류 | 결과 판정과 P50/P95/P99 |
| 5장: 가지치기·구조 변경 | 3.6절, 6절: 희소성 한계와 폭 축소 | sparse 및 0.5배 모델 비교 |
| 6장: 양자화 | 3.6절, 7~8절: 보정, QDQ/INT8, 정확도 손실 측정 | INT8 모델의 지연·크기·Top-1 비교 |

권장 실행 순서는 FP32 변환·검증 → 실시간 FP32 실행 → sparse와 0.5배 모델 생성 → INT8 보정·변환 → 네 모델의 같은 조건 비교 → 별도 검증 자료의 정확도 계산이다. 매 비교에서 입력 크기·전처리·스레드 수·워밍업·냉각 상태를 맞추고, 모델 호출 지연뿐 아니라 처리 FPS·프레임 나이·CPU·메모리·온도도 기록한다. 출력이 달라졌다고 곧바로 정확도 저하로 단정하지 말고 라벨이 있는 검증 결과를 확인한다.

학습을 마치면 다음 질문에 답할 수 있어야 한다.

1. 처리 FPS가 높아도 프레임 나이가 크면 왜 반응성이 나쁠 수 있는가?  
   대기열이나 처리 지연으로 오래된 장면을 표시할 수 있기 때문이다. 최신 프레임 우선 처리와 오래된 결과 보류를 함께 확인한다.
2. 비구조적 가지치기로 가중치 30%를 0으로 만들면 왜 실행 시간도 30% 줄었다고 할 수 없는가?  
   텐서 크기가 그대로이고 밀집 커널은 0도 계산할 수 있으므로 실제 희소 커널과 장치에서 측정해야 한다.
3. INT8 변환 성공만으로 품질을 주장할 수 없는 이유는 무엇인가?  
   보정 범위가 실제 입력을 대표하는지, 별도 라벨 검증 자료에서 정확도가 유지되는지 확인해야 한다.
4. 모델 호출 시간과 실시간 응답 시간은 어떻게 다른가?  
   `session.run()`은 모델 호출 구간만 측정한다. 프레임 획득·대기·전처리·표시까지 포함한 프레임 나이와 FPS를 별도로 측정한다.
