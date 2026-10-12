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

## 과제
앞서 작성한 MediaPipe 손동작(줌/회전) 이벤트가 발생할 때마다 MQTT로 특정 토픽(명령)을 발행하도록 코드를 통합하려면 어떻게 해야 할까?
