Python용 MQTT 클라이언트인 paho-mqtt를 설치하고, 별도의 서버 구축 없이 즉시 테스트할 수 있는 무료 공용 브로커(broker.emqx.io)를 활용한다.

## Paho-MQTT 라이브러리 설치
최근 배포된 paho-mqtt 2.0 이상 버전은 콜백 API 규격이 변경되었다.

```sh
pip install paho-mqtt
```

## 토픽 발행 (Publisher)
단방향으로 센서 데이터나 제어 명령을 브로커로 전송(Publish)한다.
```python
# mqtt_publisher.py
import paho.mqtt.client as mqtt
import time
import json

# 브로커 설정
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "rpi5/sensor/data"

# 클라이언트 생성 (paho-mqtt v2 규격)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, PORT)

print(f"[{BROKER}] 연결 완료. 데이터 발행을 시작합니다...")

try:
    count = 1
    while True:
        # 전송할 데이터 구성 (JSON 형태)
        payload = json.dumps({"count": count, "status": "running"})
        
        # 토픽 발행
        result = client.publish(TOPIC, payload)
        
        if result[0] == 0:
            print(f"발행 성공 -> 토픽: {TOPIC}, 메시지: {payload}")
            
        count += 1
        time.sleep(2)  # 2초 대기
        
except KeyboardInterrupt:
    print("\n발행을 종료합니다.")
finally:
    client.disconnect()
```

## 토픽 구독 (Subscriber)
발행자가 보낸 메시지를 실시간으로 수신(Subscribe)하고 대기한다.
```python
# mqtt_subscriber.py
import paho.mqtt.client as mqtt

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "rpi5/sensor/data"

# 브로커 접속 완료 시 실행되는 콜백 함수
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[{BROKER}] 연결 성공!")
        # 연결 직후 구독 시작
        client.subscribe(TOPIC)
        print(f"[{TOPIC}] 토픽 구독 대기 중...")
    else:
        print(f"연결 실패 (코드: {reason_code})")

# 메시지 수신 시 실행되는 콜백 함수
def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8")
    print(f"[수신] 토픽: {message.topic} | 메시지: {payload}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# 콜백 함수 등록
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT)

# 네트워크 루프 무한 대기 (블로킹 함수)
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n구독을 종료합니다.")
finally:
    client.disconnect()
```

## 발행 및 구독 동시 처리 (Pub/Sub)
클라이언트가 메시지를 수신하기 위해 백그라운드에서 대기하면서, 동시에 주기적으로 자신의 메시지를 발행한다.
```python
# mqtt_pub_sub.py
import paho.mqtt.client as mqtt
import time

BROKER = "broker.emqx.io"
PORT = 1883
PUB_TOPIC = "rpi5/status"
SUB_TOPIC = "server/command"

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("브로커 연결 성공!")
        client.subscribe(SUB_TOPIC)
        print(f"[{SUB_TOPIC}] 구독 시작.")

def on_message(client, userdata, message):
    msg = message.payload.decode("utf-8")
    print(f"\n[명령 수신] {message.topic}: {msg}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT)

# 구독 및 네트워크 처리를 백그라운드 스레드로 실행 (논블로킹)
client.loop_start()

print("메인 루프에서 데이터 발행을 시작합니다. (종료: Ctrl+C)")
try:
    while True:
        timestamp = time.strftime("%H:%M:%S")
        msg = f"Device active at {timestamp}"
        client.publish(PUB_TOPIC, msg)
        print(f"[발행] {PUB_TOPIC}: {msg}")
        
        # 다른 터미널에서 구독자 스크립트나 앱으로 server/command 토픽에 메시지를 
        # 발행하면 on_message가 백그라운드에서 가로채어 출력합니다.
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n종료합니다.")
finally:
    client.loop_stop()  # 백그라운드 스레드 종료
    client.disconnect()
```

## 과제1

### 참여자 및 토픽 구조
- 팀 및 인원: 5개 조(team1 ~ team5), 조당 6명(mem1 ~ mem6)
- 토픽 규칙
  - 제어 신호: game/control (강사 발행: START, STOP)
  - 플레이어 메시지 발행 경로: game/team/{target_team}/{target_mem}
    - 예: 1조 2번(team1/mem2)이 3조 4번에게 메시지를 보낼 때: game/team/team3/mem4로 발행
    - 수신자는 자신이 속한 토픽(game/team/{my_team}/{my_mem}) 구독
  - 채점 이벤트 보고: game/score/report (강사 전용 수신 토픽)

### 게임 규칙 및 채점 메커니즘
1. 강사의 시작 신호: 강사가 game/control로 START 메시지를 발행하면 게임이 시작됩니다.
2, 도전(Ping)과 응답(Pong)
   - 송신자 A가 수신자 B에게 유효한 챌린지 메시지({"from_team": "...", "from_mem": "...", "msg_id": "...", "type": "PING"})를 전송합니다.
   - 제한 시간(예: 3초) 내에 B가 A에게 응답(PONG)을 보내면 성공(0점)으로 처리됩니다.
   - B가 타임아웃 내에 응답하지 못하면 A가 속한 팀에 +1점이 부여됩니다.
3. 강사의 종료 신호 및 집계
   - 강사가 STOP 신호를 발행하면 모든 상호작용이 정지되고, 강사가 팀별 합산 점수를 출력합니다.

### 학생 배포 코드

다음 코드를 게임 규칙에 맞게 구현한다.

```python
import json
import random
import threading
import time
import uuid
import paho.mqtt.client as mqtt

# ==========================================
# 1. 설정값 (본인의 조와 번호에 맞게 수정)
# ==========================================
MY_TEAM = "team1"     # 본인 조: team1 ~ team5
MY_MEM = "mem1"       # 본인 식별자: mem1 ~ mem6
BROKER = "localhost"  # 강사 브로커 IP
PORT = 1883

MY_TOPIC = f"game/team/{MY_TEAM}/{MY_MEM}"
CONTROL_TOPIC = "game/control"
SCORE_TOPIC = "game/score/report"

game_running = False
pending_requests = {}  # {msg_id: {"sent_time": float, "target_team": str, "target_mem": str}}
lock = threading.Lock()


def on_connect(client, userdata, flags, rc, properties=None):
    """브로커 연결 성공 시 구독(Subscribe) 설정"""
    print(f"[{MY_TEAM}/{MY_MEM}] 브로커 연결 성공. 토픽 구독을 시작합니다.")
    # 내 개인 토픽과 게임 제어 토픽 구독
    client.subscribe([(MY_TOPIC, 1), (CONTROL_TOPIC, 1)])


def on_message(client, userdata, msg):
    """메시지 수신 시 호출되는 콜백 함수"""
    global game_running
    payload_str = msg.payload.decode()

    # 강사의 시작/종료 제어 신호 처리
    if msg.topic == CONTROL_TOPIC:
        if payload_str == "START":
            print("[신호] 게임 시작!")
            game_running = True
        elif payload_str == "STOP":
            print("[신호] 게임 종료!")
            game_running = False
        return

    # [미션 1] 상대방이 보낸 메시지 처리 로직 구현
    if msg.topic == MY_TOPIC and game_running:
        try:
            packet = json.loads(payload_str)
            p_type = packet.get("type")
            sender_team = packet.get("from_team")
            sender_mem = packet.get("from_mem")
            msg_id = packet.get("msg_id")

            # -------------------------------------------------------------
            # TODO 1-1: 상대방이 나에게 "PING"을 보냈을 때
            # - 상대방의 토픽(game/team/{sender_team}/{sender_mem})으로 즉시 응답
            # - 패킷 구성: {"type": "PONG", "from_team": MY_TEAM, "from_mem": MY_MEM, "msg_id": msg_id}
            # - client.publish(...) 사용
            # -------------------------------------------------------------
            if p_type == "PING":
                pass  # 여기에 코드 작성

            # -------------------------------------------------------------
            # TODO 1-2: 내가 보낸 공격에 대해 상대가 "PONG"으로 응답했을 때
            # - pending_requests 딕셔너리에서 해당 msg_id를 제거하여 0점 처리 (방어 성공)
            # - 스레드 안전을 위해 `with lock:` 블록 안에서 처리할 것
            # -------------------------------------------------------------
            elif p_type == "PONG":
                pass  # 여기에 코드 작성

        except Exception as e:
            print(f"[패킷 파싱 오류] {e}")


def timeout_checker(client):
    """
    [미션 2] 타임아웃 검사 및 득점 보고 스레드
    내가 보낸 PING에 대해 상대방이 제한시간(3초) 내에 응답하지 않으면 득점 처리
    """
    TIMEOUT_SEC = 3.0
    while True:
        if game_running:
            now = time.time()
            expired_ids = []

            # -------------------------------------------------------------
            # TODO 2-1: pending_requests 순회하며 TIMEOUT_SEC가 지난 메시지 찾기
            # - 경과 시간: (now - info["sent_time"])
            # - 시간이 초과된 항목은 pending_requests에서 제거하고 expired_ids에 기록
            # -------------------------------------------------------------
            with lock:
                pass  # 여기에 코드 작성

            # -------------------------------------------------------------
            # TODO 2-2: 응답이 없는 상대에 대해 강사 서버(SCORE_TOPIC)로 +1점 득점 보고
            # - 보고 패킷 구성:
            #   {
            #       "winner_team": MY_TEAM,
            #       "winner_mem": MY_MEM,
            #       "reason": f"No response from {info['target_team']}/{info['target_mem']}"
            #   }
            # - client.publish(...) 사용
            # -------------------------------------------------------------
            for msg_id, info in expired_ids:
                pass  # 여기에 코드 작성

        time.sleep(0.5)


def attack_loop(client):
    """
    [미션 3] 다른 팀 구성원을 무작위로 지목하여 핑(PING) 공격 전송
    """
    TEAMS = [f"team{i}" for i in range(1, 6)]
    MEMBERS = [f"mem{i}" for i in range(1, 7)]

    while True:
        if game_running:
            # -------------------------------------------------------------
            # TODO 3-1: 상대 팀 및 상대 팀원 무작위 선정
            # - 우리 팀(MY_TEAM)을 제외한 팀 중 1팀 선택
            # - 멤버(MEMBERS) 중 1명 선택
            # - 목표 토픽: f"game/team/{target_team}/{target_mem}"
            # -------------------------------------------------------------
            target_team = None  # 코드 작성
            target_mem = None   # 코드 작성
            target_topic = None # 코드 작성

            # 고유 메시지 ID 생성
            msg_id = str(uuid.uuid4())

            # -------------------------------------------------------------
            # TODO 3-2: PING 메시지 패킷 구성 및 발행
            # - 패킷: {"type": "PING", "from_team": MY_TEAM, "from_mem": MY_MEM, "msg_id": msg_id}
            # - pending_requests에 기록 (보낸 시간, 대상 팀, 대상 멤버)
            # - client.publish(target_topic, ...)
            # -------------------------------------------------------------
            
            # 발행 주기 제어 (네트워크 부하 방지용 1~2.5초 대기)
            time.sleep(random.uniform(1.0, 2.5))
        else:
            time.sleep(1)


# ==========================================
# 실행부 (수정 불필요)
# ==========================================
if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()

        # 백그라운드 스레드 실행
        threading.Thread(target=timeout_checker, args=(client,), daemon=True).start()
        threading.Thread(target=attack_loop, args=(client,), daemon=True).start()

        print(f"[{MY_TEAM}/{MY_MEM}] 준비 완료. 강사의 START 신호를 대기합니다...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        client.loop_stop()
        client.disconnect()
```


## 과제2
앞서 작성한 MediaPipe 손동작(줌/회전) 이벤트가 발생할 때마다 MQTT로 특정 토픽(명령)을 발행하도록 코드를 통합하려면 어떻게 해야 할까?
