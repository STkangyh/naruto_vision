# Video Recorder + Naruto Jutsu

OpenCV와 MediaPipe를 이용한 웹캠/카메라 영상 녹화 + 나루토 기술 인식 프로그램

## 기능 설명

### 주요 기능

1. **실시간 카메라 영상 표시**
   - OpenCV의 `cv.VideoCapture`를 사용하여 웹캠 영상을 실시간으로 캡처
   - 좌우 반전(거울 모드)으로 자연스러운 화면 표시

2. **동영상 녹화**
   - OpenCV의 `cv.VideoWriter`를 사용하여 영상을 동영상 파일로 저장
   - XVID 코덱을 사용한 AVI 형식으로 저장
   - 녹화 파일은 `recordings` 폴더에 타임스탬프와 함께 저장
   - 파일명 형식: `recording_YYYYMMDD_HHMMSS.avi`

3. **Preview 모드**
   - 카메라 영상을 확인만 하고 녹화하지 않는 모드
   - 화면 좌측 상단에 "PREVIEW" 텍스트 표시 (녹색)
   - 화면 하단에 키 조작 안내 표시

4. **Record 모드**
   - 카메라 영상을 파일로 저장하는 모드
   - 화면 좌측 상단에 빨간 원(●) 표시
   - "REC" 텍스트와 "Recording..." 메시지 표시 (빨간색)

5. **🥷 나루토 기술 인식 모드 (Naruto Jutsu)**
   - MediaPipe HandLandmarker로 손 관절 21개를 실시간 추적
   - 손 제스처를 분석하여 6가지 나루토 기술을 발동
   - 각 기술마다 고유한 파티클 + 애니메이션 이펙트 표시
   - 녹화 중에도 이펙트가 영상에 함께 저장됨

---

### 🥷 나루토 기술 목록

| 제스처 | 기술 | 이펙트 |
|--------|------|--------|
| ✊ 주먹 (모든 손가락 접힘) | 螺旋丸 **RASENGAN** | 회전하는 파란 나선 구체 |
| ✌️ 검지 + 중지 (V사인) | 影分身 **KAGE BUNSHIN** | 흰 연기 + 분신 효과 |
| 🖐️ 손 활짝 (5손가락 모두) | 火遁・豪火球 **KATON** | 빨간-주황 불꽃 이펙트 |
| ☝️ 검지만 핌 | 雷切 **RAIKIRI** | 방사형 번개 이펙트 |
| 🤙 엄지 + 새끼손가락 | 風遁・螺旋手裏剣 **FUTON** | 회전하는 초록 나선 바람 |
| 👆 엄지 + 검지 (총 모양) | チャクラ集中 **CHAKRA FOCUS** | 보라 육각형 차크라 오라 |

---

### 키 조작

| 키 | 동작 |
|----|------|
| `SPACE` | Preview ↔ Record 모드 전환 |
| `N` | 나루토 모드 ON/OFF 토글 |
| `ESC` | 프로그램 종료 |

---

## 설치 방법

### 필수 요구사항

- Python 3.10 이상
- OpenCV, MediaPipe, NumPy

### 패키지 설치

```bash
pip install -r requirements.txt
```

### 모델 파일 다운로드

```bash
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## 실행 방법

```bash
python3 video_recorder.py
```

## 사용 방법

1. 프로그램을 실행하면 웹캠이 자동으로 활성화됩니다.
2. 처음에는 **Preview 모드 + 나루토 모드 ON** 상태로 시작됩니다.
3. 카메라 창을 클릭하여 활성화한 후 키 입력이 가능합니다.
4. 카메라 앞에서 위의 제스처를 취하면 기술 이펙트가 발동됩니다.
5. **SPACE 키**를 눌러 녹화를 시작/중지합니다.
6. **ESC 키**를 눌러 프로그램을 종료합니다.

---

## 저장 위치

녹화된 동영상은 프로그램 실행 디렉토리의 `recordings` 폴더에 저장됩니다.

---

## 기술 세부사항

### 클래스 구조

#### `VideoRecorder` (`video_recorder.py`)
- `initialize_camera()`: 카메라 초기화 및 설정
- `start_recording()` / `stop_recording()`: 녹화 제어
- `draw_recording_indicator()`: Record 모드 UI
- `draw_preview_indicator()`: Preview 모드 UI
- `run()`: 메인 루프 (손 인식 → 이펙트 → 모드 표시 → 녹화)

#### `NarutoJutsuRecognizer` (`naruto_jutsu.py`)
- MediaPipe `HandLandmarker` (Tasks API, LIVE_STREAM 모드) 사용
- `process_frame(frame)`: 손 감지 → 제스처 인식 → 이펙트 렌더링
- `toggle()`: 나루토 모드 ON/OFF

#### `JutsuEffect` (`naruto_jutsu.py`)
- 6가지 기술별 애니메이션 이펙트 렌더링
- `Particle` 파티클 시스템으로 불꽃·번개·연기 효과 구현

#### `detect_gesture(lm)` (`naruto_jutsu.py`)
- 손가락 21개 랜드마크를 분석하여 제스처 판별

### 설정 값

- 기본 해상도: 640×480
- FPS: 20 (녹화), 30 (감지 타임스탬프)
- 코덱: XVID / AVI
- 제스처 안정화: 8프레임 버퍼 다수결

### 파일 구조

```
recorder/
├── video_recorder.py       # 메인 프로그램
├── naruto_jutsu.py         # 나루토 기술 인식 & 이펙트 모듈
├── hand_landmarker.task    # MediaPipe 손 인식 모델
├── requirements.txt        # 의존성 목록
├── README.md               # 이 파일
└── recordings/             # 녹화 파일 저장 폴더 (자동 생성)
    └── recording_YYYYMMDD_HHMMSS.avi
```

---

## 문제 해결

### 카메라를 열 수 없는 경우
- 다른 프로그램이 카메라를 사용 중인지 확인
- macOS 시스템 환경설정 → 개인 정보 보호 → 카메라 권한 확인

### 나루토 기술이 인식되지 않는 경우
- 조명이 충분한 환경에서 시도
- 손 전체가 카메라에 보이도록 거리 조절
- `hand_landmarker.task` 파일이 프로젝트 폴더에 있는지 확인

## 라이선스

MIT License


## 기능 설명

### 주요 기능

1. **실시간 카메라 영상 표시**
   - OpenCV의 `cv.VideoCapture`를 사용하여 웹캠 영상을 실시간으로 캡처
   - 화면에 현재 카메라 영상을 지속적으로 표시

2. **동영상 녹화**
   - OpenCV의 `cv.VideoWriter`를 사용하여 영상을 동영상 파일로 저장
   - XVID 코덱을 사용한 AVI 형식으로 저장
   - 녹화 파일은 `recordings` 폴더에 타임스탬프와 함께 저장
   - 파일명 형식: `recording_YYYYMMDD_HHMMSS.avi`

3. **Preview 모드**
   - 카메라 영상을 확인만 하고 녹화하지 않는 모드
   - 화면 좌측 상단에 "PREVIEW" 텍스트 표시 (녹색)
   - 화면 하단에 키 조작 안내 표시

4. **Record 모드**
   - 카메라 영상을 파일로 저장하는 모드
   - 화면 좌측 상단에 빨간 원(●) 표시
   - "REC" 텍스트와 "Recording..." 메시지 표시 (빨간색)
   - 실제 프레임을 동영상 파일에 저장

### 키 조작

- **SPACE 키**: Preview 모드와 Record 모드 전환
  - Preview 모드에서 SPACE 누르면 → 녹화 시작 (Record 모드)
  - Record 모드에서 SPACE 누르면 → 녹화 중지 (Preview 모드)
  
- **ESC 키**: 프로그램 종료
  - 녹화 중인 경우 자동으로 저장 후 종료

## 설치 방법

### 필수 요구사항

- Python 3.6 이상
- OpenCV 라이브러리

### 설치

```bash
pip install opencv-python numpy
```

## 실행 방법

```bash
python video_recorder.py
```

## 사용 방법

1. 프로그램을 실행하면 웹캠이 자동으로 활성화됩니다.
2. 처음에는 **Preview 모드**로 시작됩니다.
3. **SPACE 키**를 눌러 녹화를 시작합니다.
4. 녹화 중에는 화면 좌측 상단에 빨간 원과 "REC" 표시가 나타납니다.
5. 다시 **SPACE 키**를 눌러 녹화를 중지합니다.
6. **ESC 키**를 눌러 프로그램을 종료합니다.

## 저장 위치

녹화된 동영상은 프로그램 실행 디렉토리의 `recordings` 폴더에 저장됩니다.

## 기술 세부사항

### 클래스 구조

#### VideoRecorder 클래스

주요 메서드:
- `initialize_camera()`: 카메라 초기화 및 설정
- `start_recording()`: 녹화 시작 및 VideoWriter 초기화
- `stop_recording()`: 녹화 중지 및 파일 저장
- `draw_recording_indicator()`: Record 모드 UI 표시
- `draw_preview_indicator()`: Preview 모드 UI 표시
- `run()`: 메인 루프 실행
- `cleanup()`: 리소스 정리

### 설정 값

- 기본 해상도: 640x480
- FPS: 20
- 코덱: XVID
- 파일 형식: AVI

### 파일 구조

```
recorder/
├── video_recorder.py    # 메인 프로그램
├── README.md           # 이 파일
└── recordings/         # 녹화 파일 저장 폴더 (자동 생성)
    └── recording_YYYYMMDD_HHMMSS.avi
```

## 문제 해결

### 카메라를 열 수 없는 경우
- 다른 프로그램이 카메라를 사용 중인지 확인
- 카메라가 올바르게 연결되어 있는지 확인
- macOS의 경우 시스템 환경설정에서 카메라 권한 확인

### 녹화 파일이 생성되지 않는 경우
- `recordings` 폴더의 쓰기 권한 확인
- 디스크 공간 확인

## 라이선스

MIT License

## 작성자

Video Recorder v1.0
