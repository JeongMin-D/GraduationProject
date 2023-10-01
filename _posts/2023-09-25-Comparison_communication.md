---
title: "Comparison Communication"
date: 2023-09-25
categories: jekyll update
---

# 통신 프로토콜(데이터베이스 포함) 설명 및 비교

## 통신 안정성, 실시간 성능, 통신 속도 비교(파이썬 예제 포함)

### 1. TCP (Transmission Control Protocol):
- TCP는 네트워크를 통해 안정적이고 순서 있으며 오류 확인된 데이터 전달을 제공하는 연결 지향 프로토콜입니다.

```python
import socket

# Server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(5)

(client_socket, client_address) = server_socket.accept()
data = client_socket.recv(1024)
print(f"Received data: {data.decode()}")

# Client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12345))
client_socket.sendall(b'Hello, TCP Server!')
```

### 2. UDP (User Datagram Protocol):
- UDP는 전달을 보장하지 않지만 더 빠른 전송을 제공하는 비연결 프로토콜입니다.

```python
import socket

# Server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('localhost', 12345))

data, client_address = server_socket.recvfrom(1024)
print(f"Received data: {data.decode()}")

# Client
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.sendto(b'Hello, UDP Server!', ('localhost', 12345))
```

### 3. Websockets:
- 웹소켓은 표준 HTTP 포트를 통해 작동하는 단일 장기 연결을 통해 전이중 통신 채널을 제공합니다.

```python
import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)

asyncio.get_event_loop().run_until_complete(
    websockets.serve(echo, 'localhost', 8765))
asyncio.get_event_loop().run_forever()
```

### 4. MQTT (Message Queuing Telemetry Transport):
- MQTT는 대기 시간이 길거나 신뢰할 수 없는 네트워크에 최적화된 소형 센서 및 모바일 장치를 위한 경량 메시징 프로토콜입니다.

```python
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    client.subscribe("topic/test")

def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("broker.example.com", 1883, 60)
client.loop_start()

client.publish("topic/test", "Hello, MQTT!")
```

### 5. MySQL:
- MySQL은 데이터 관리 및 조작을 위해 구조화된 쿼리 언어(SQL)를 사용하는 인기 있는 관계형 데이터베이스 관리 시스템입니다.

```python
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="username",
    password="password",
    database="mydatabase"
)

cursor = db.cursor()
cursor.execute("CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255))")
cursor.execute("INSERT INTO users (name) VALUES ('John')")
db.commit()

cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)
```

### 6. NoSQL:
- NoSQL 데이터베이스는 전통적인 관계형 데이터베이스 관리 시스템(RDBMS) 구조에 의존하지 않는 데이터베이스 범주입니다.

```python
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["mycollection"]

data = {"name": "John", "age": 30}
collection.insert_one(data)

result = collection.find({"name": "John"})
for doc in result:
    print(doc)

```

## 비교:

### 통신 안정성
- TCP는 안정적이고 순차적인 데이터 전달로 인해 높은 통신 안정성을 제공합니다.
- UDP는 전달을 보장하지 않기 때문에 속도를 위해 안정성을 희생합니다.

### 실시간성
- 웹소켓과 MQTT는 실시간 통신용으로 설계되어 대기 시간이 짧은 양방향 통신을 제공합니다.

### 통신 속도
- UDP는 연결이 없는 특성과 핸드셰이크 절차가 없기 때문에 가장 빠릅니다.
- 웹소켓과 MQTT도 빠르지만 UDP에 비해 오버헤드가 약간 더 높을 수 있습니다.
- TCP는 연결을 설정하고 안정성을 보장하므로 UDP에 비해 속도가 느립니다.

### 프로토콜/데이터베이스 선택은 특정 사용 사례, 요구 사항, 속도, 안정성 및 기타 요소 간의 균형에 따라 달라집니다.