import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math
import os

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles


# ───────────────────────────────────────────────
#  손가락 상태 감지 유틸
# ───────────────────────────────────────────────

# MediaPipe Hand Landmark 인덱스
WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20
THUMB_MCP   = 2
INDEX_MCP   = 5
MIDDLE_MCP  = 9
RING_MCP    = 13
PINKY_MCP   = 17
INDEX_PIP   = 6
MIDDLE_PIP  = 10
RING_PIP    = 14
PINKY_PIP   = 18


def _tip_above_mcp(lm, tip_idx, mcp_idx):
    """손가락 끝이 MCP 관절보다 위에 있으면 True (손가락 펴짐)."""
    return lm[tip_idx].y < lm[mcp_idx].y


def _distance(lm, a, b):
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return math.sqrt(dx * dx + dy * dy)


def count_fingers(lm):
    """펴진 손가락 수 반환 (엄지 제외 4개 + 엄지 1개)."""
    count = 0
    # 엄지: x축 기준
    if lm[THUMB_TIP].x < lm[THUMB_MCP].x:   # 오른손 기준
        count += 1
    for tip, mcp in [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                     (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]:
        if _tip_above_mcp(lm, tip, mcp):
            count += 1
    return count


# ───────────────────────────────────────────────
#  제스처 인식
# ───────────────────────────────────────────────

def detect_gesture(lm):
    """
    손 랜드마크에서 나루토 기술 제스처를 판별한다.

    반환값:
        str | None  – 감지된 기술 이름 또는 None
    """
    # 각 손가락 펴짐 여부
    thumb_up   = lm[THUMB_TIP].x  < lm[THUMB_MCP].x
    index_up   = _tip_above_mcp(lm, INDEX_TIP,  INDEX_MCP)
    middle_up  = _tip_above_mcp(lm, MIDDLE_TIP, MIDDLE_MCP)
    ring_up    = _tip_above_mcp(lm, RING_TIP,   RING_MCP)
    pinky_up   = _tip_above_mcp(lm, PINKY_TIP,  PINKY_MCP)

    fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    n_up = sum(fingers)

    # ── 라센간: 주먹 (모든 손가락 접힘) ──────────────
    if n_up == 0:
        return "RASENGAN"

    # ── 차크라 집중: 엄지+검지만 핀 (총기 모양) ──────
    if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "CHAKRA_FOCUS"

    # ── 카게부신: 검지+중지 (V/평화) ─────────────────
    if not thumb_up and index_up and middle_up and not ring_up and not pinky_up:
        return "KAGE_BUNSHIN"

    # ── 화둔: 손 활짝 핌 (5 손가락 모두) ─────────────
    if n_up == 5:
        return "KATON"

    # ── 뇌둔: 검지만 핌 ──────────────────────────────
    if not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "RAIKIRI"

    # ── 풍둔: 소지(새끼)+엄지 (shaka 🤙) ─────────────
    if thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
        return "FUTON"

    return None


# ───────────────────────────────────────────────
#  파티클 시스템 (효과 연출용)
# ───────────────────────────────────────────────

class Particle:
    def __init__(self, x, y, color, speed=3, size=4, lifetime=30):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.age = 0
        angle = np.random.uniform(0, 2 * math.pi)
        spd   = np.random.uniform(speed * 0.5, speed * 1.5)
        self.vx = math.cos(angle) * spd
        self.vy = math.sin(angle) * spd

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        # 중력
        self.vy += 0.15
        self.size = max(1, self.size - 0.1)

    @property
    def alive(self):
        return self.age < self.lifetime

    def draw(self, frame):
        alpha = 1.0 - self.age / self.lifetime
        r, g, b = self.color
        cv.circle(frame, (int(self.x), int(self.y)), max(1, int(self.size)),
                  (int(b * alpha), int(g * alpha), int(r * alpha)), -1)


# ───────────────────────────────────────────────
#  효과 렌더러
# ───────────────────────────────────────────────

class JutsuEffect:
    """
    현재 활성화된 기술 하나를 화면에 그린다.
    여러 기술이 겹칠 수 없도록 단일 인스턴스로 사용한다.
    """

    JUTSU_INFO = {
        "RASENGAN":      {"name": "螺旋丸  RASENGAN",      "color": (0, 191, 255),   "icon": "🌀"},
        "KAGE_BUNSHIN":  {"name": "影分身  KAGE BUNSHIN",  "color": (255, 255,   0), "icon": "✌️"},
        "KATON":         {"name": "火遁・豪火球  KATON",   "color": (0,  80, 255),   "icon": "🔥"},
        "RAIKIRI":       {"name": "雷切  RAIKIRI",         "color": (255, 255, 100), "icon": "⚡"},
        "FUTON":         {"name": "風遁・螺旋手裏剣  FUTON","color": (144, 238, 144), "icon": "🌪️"},
        "CHAKRA_FOCUS":  {"name": "チャクラ集中  CHAKRA",  "color": (138,  43, 226), "icon": "👆"},
    }

    def __init__(self):
        self.active_jutsu  = None
        self.particles: list[Particle] = []
        self.frame_count   = 0          # 기술 활성화 이후 프레임 수
        self.palm_center   = (320, 240)
        self._spin_angle   = 0.0        # 나선 회전용

    # ── 외부 인터페이스 ────────────────────────────────

    def activate(self, jutsu_name: str, palm_center: tuple):
        """기술 활성화 (같은 기술이면 유지, 다른 기술이면 교체)."""
        if jutsu_name != self.active_jutsu:
            self.active_jutsu = jutsu_name
            self.frame_count  = 0
            self.particles.clear()
            self._spin_angle  = 0.0
        self.palm_center = palm_center

    def deactivate(self):
        """기술 비활성화."""
        self.active_jutsu = None
        self.particles.clear()
        self.frame_count  = 0

    def update_and_draw(self, frame):
        """매 프레임 호출 – 파티클 업데이트 & 렌더링."""
        if self.active_jutsu is None:
            return

        info  = self.JUTSU_INFO[self.active_jutsu]
        color = info["color"]
        cx, cy = self.palm_center
        h, w   = frame.shape[:2]

        self._spawn_particles(cx, cy, color)

        # 파티클 업데이트 & 그리기
        self.particles = [p for p in self.particles if p.alive]
        for p in self.particles:
            p.update()
            p.draw(frame)

        # 기술별 메인 효과
        self._draw_main_effect(frame, cx, cy, color)

        # HUD: 기술 이름
        self._draw_hud(frame, info)

        self.frame_count  += 1
        self._spin_angle  += 5.0

    # ── 파티클 생성 ────────────────────────────────────

    def _spawn_particles(self, cx, cy, color):
        jutsu = self.active_jutsu
        if jutsu == "RASENGAN":
            for _ in range(8):
                self.particles.append(Particle(cx, cy, color, speed=5, size=6, lifetime=25))
        elif jutsu == "KATON":
            for _ in range(12):
                fire_color = (
                    np.random.randint(200, 255),
                    np.random.randint(50, 150),
                    0
                )
                self.particles.append(Particle(cx, cy, fire_color, speed=6, size=8, lifetime=20))
        elif jutsu == "RAIKIRI":
            for _ in range(6):
                self.particles.append(Particle(cx, cy, (255, 255, 180), speed=8, size=4, lifetime=15))
        elif jutsu == "FUTON":
            for _ in range(5):
                self.particles.append(Particle(cx, cy, color, speed=7, size=5, lifetime=22))
        elif jutsu == "KAGE_BUNSHIN":
            for _ in range(4):
                smoke = (200, 200, 200)
                self.particles.append(Particle(cx, cy, smoke, speed=3, size=10, lifetime=35))
        elif jutsu == "CHAKRA_FOCUS":
            for _ in range(3):
                self.particles.append(Particle(cx, cy, color, speed=4, size=5, lifetime=28))

    # ── 기술별 메인 이펙트 ─────────────────────────────

    def _draw_main_effect(self, frame, cx, cy, color):
        jutsu = self.active_jutsu
        t     = self.frame_count
        angle = self._spin_angle
        bgr   = (color[2], color[1], color[0])  # RGB→BGR

        if jutsu == "RASENGAN":
            self._draw_rasengan(frame, cx, cy, bgr, angle)

        elif jutsu == "KATON":
            self._draw_katon(frame, cx, cy, t)

        elif jutsu == "RAIKIRI":
            self._draw_raikiri(frame, cx, cy, t)

        elif jutsu == "FUTON":
            self._draw_futon(frame, cx, cy, bgr, angle)

        elif jutsu == "KAGE_BUNSHIN":
            self._draw_kage_bunshin(frame, cx, cy, bgr, t)

        elif jutsu == "CHAKRA_FOCUS":
            self._draw_chakra_focus(frame, cx, cy, bgr, t)

    # ─── 라센간 ─────────────────────────────────────────
    def _draw_rasengan(self, frame, cx, cy, bgr, angle):
        overlay = frame.copy()
        # 바깥 나선 고리
        for r in range(60, 20, -10):
            alpha_circle = 0.3
            cv.circle(overlay, (cx, cy), r, bgr, 2)
        cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 회전하는 궤도 점
        for i in range(12):
            a = math.radians(angle + i * 30)
            r = 45
            px = int(cx + r * math.cos(a))
            py = int(cy + r * math.sin(a))
            cv.circle(frame, (px, py), 4, bgr, -1)

        # 중심 코어
        cv.circle(frame, (cx, cy), 18, bgr, -1)
        cv.circle(frame, (cx, cy), 22, (255, 255, 255), 2)

    # ─── 화둔 ────────────────────────────────────────────
    def _draw_katon(self, frame, cx, cy, t):
        overlay = frame.copy()
        # 여러 크기 불 원
        for i, (r, a) in enumerate([(70, 0.4), (50, 0.5), (30, 0.6)]):
            flicker = int(5 * math.sin(t * 0.3 + i))
            cv.circle(overlay, (cx, cy), r + flicker,
                      (0, int(60 + i*40), 255 - i*40), -1)
        cv.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        # 중심 밝은 코어
        cv.circle(frame, (cx, cy), 15, (0, 200, 255), -1)

    # ─── 뇌절 ────────────────────────────────────────────
    def _draw_raikiri(self, frame, cx, cy, t):
        overlay = frame.copy()
        # 번개 방사형
        np.random.seed(t % 30)
        for _ in range(10):
            angle = np.random.uniform(0, 2 * math.pi)
            length = np.random.randint(40, 90)
            segs   = 5
            prev_x, prev_y = cx, cy
            for s in range(segs):
                ratio = (s + 1) / segs
                base_x = cx + int(length * ratio * math.cos(angle))
                base_y = cy + int(length * ratio * math.sin(angle))
                jitter_x = base_x + np.random.randint(-8, 8)
                jitter_y = base_y + np.random.randint(-8, 8)
                cv.line(overlay, (prev_x, prev_y), (jitter_x, jitter_y),
                        (255, 255, 200), 2)
                prev_x, prev_y = jitter_x, jitter_y
        cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        # 코어
        cv.circle(frame, (cx, cy), 14, (200, 220, 255), -1)

    # ─── 풍둔 ────────────────────────────────────────────
    def _draw_futon(self, frame, cx, cy, bgr, angle):
        overlay = frame.copy()
        # 회전하는 나선형 링들
        for ring in range(3):
            r = 35 + ring * 20
            for i in range(16):
                a  = math.radians(angle * (1 + ring * 0.3) + i * 22.5)
                px = int(cx + r * math.cos(a))
                py = int(cy + r * math.sin(a))
                cv.circle(overlay, (px, py), 3, bgr, -1)
        cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv.circle(frame, (cx, cy), 16, bgr, -1)

    # ─── 카게부신 ─────────────────────────────────────────
    def _draw_kage_bunshin(self, frame, cx, cy, bgr, t):
        overlay = frame.copy()
        # 연기 효과 (흰 반투명 원)
        for i in range(3):
            offset_x = int(40 * math.cos(math.radians(t * 3 + i * 120)))
            offset_y = int(20 * math.sin(math.radians(t * 3 + i * 120)))
            r = 30 + int(5 * math.sin(t * 0.2 + i))
            cv.circle(overlay, (cx + offset_x, cy + offset_y), r,
                      (220, 220, 220), -1)
        cv.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        # 중심 파란 도장 원
        cv.circle(frame, (cx, cy), 16, bgr, -1)
        cv.circle(frame, (cx, cy), 20, (255, 255, 255), 2)

    # ─── 차크라 집중 ──────────────────────────────────────
    def _draw_chakra_focus(self, frame, cx, cy, bgr, t):
        overlay = frame.copy()
        pulse = int(10 * abs(math.sin(t * 0.1)))
        # 바깥 아우라 링
        for r in range(50 + pulse, 20, -12):
            cv.circle(overlay, (cx, cy), r, bgr, 1)
        cv.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        # 육각형 모양
        for i in range(6):
            a1 = math.radians(t * 2 + i * 60)
            a2 = math.radians(t * 2 + (i + 1) * 60)
            p1 = (int(cx + 35 * math.cos(a1)), int(cy + 35 * math.sin(a1)))
            p2 = (int(cx + 35 * math.cos(a2)), int(cy + 35 * math.sin(a2)))
            cv.line(frame, p1, p2, bgr, 2)
        cv.circle(frame, (cx, cy), 12, bgr, -1)

    # ── HUD (프레임 내부 – 비활성화, 패널로 이전) ────────
    def _draw_hud(self, frame, info):
        pass  # 기술 정보는 하단 패널에 표시


# ───────────────────────────────────────────────
#  메인 인식기 클래스
# ───────────────────────────────────────────────

class NarutoJutsuRecognizer:
    """
    MediaPipe Hands 로 손 인식 + JutsuEffect 로 기술 렌더링.
    VideoRecorder 의 메인 루프에서 process_frame() 을 호출하여 사용.
    """

    GESTURE_HOLD_FRAMES = 8   # 몇 프레임 연속으로 같은 제스처여야 활성화할지
    # 모델 파일 경로 (같은 폴더에 있다고 가정)
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.effect  = JutsuEffect()

        self._gesture_buffer: list[str | None] = []
        self._current_gesture = None

        # 최신 감지 결과 저장소 (VIDEO 모드 콜백용)
        self._latest_result = None
        self._frame_timestamp_ms = 0

        self._hand_landmarker = None
        self._draw_utils  = drawing_utils
        self._draw_styles = drawing_styles

        if self.enabled:
            self._init_landmarker()

    def _init_landmarker(self):
        """HandLandmarker 초기화 (새 Tasks API)."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {self.MODEL_PATH}")
            print("        hand_landmarker.task 파일을 프로젝트 폴더에 넣어주세요.")
            self.enabled = False
            return

        def _result_callback(result, output_image, timestamp_ms):
            self._latest_result = result

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
            result_callback=_result_callback,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(options)
        print("HandLandmarker 초기화 완료")

    # ── 공개 메서드 ────────────────────────────────────

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled and self._hand_landmarker is None:
            self._init_landmarker()
        if not self.enabled:
            # OFF 전환 시 상태 완전 초기화
            self.effect.deactivate()
            self._current_gesture = None
            self._gesture_buffer.clear()
            self._latest_result = None
        print(f"나루토 모드: {'ON' if self.enabled else 'OFF'}")

    def process_frame(self, frame):
        """
        frame 에 손 인식 + 기술 효과를 그려 반환한다.
        frame 은 BGR numpy array (이미 flip 된 상태).
        """
        if not self.enabled or self._hand_landmarker is None:
            # OFF 상태에서도 프레임 우상단에 모드 표시
            self._draw_mode_indicator(frame)
            return frame

        h, w = frame.shape[:2]

        # MediaPipe Image 변환 (RGB)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 타임스탬프 (단조 증가 필수)
        self._frame_timestamp_ms += 33   # ≈30fps
        self._hand_landmarker.detect_async(mp_image, self._frame_timestamp_ms)

        gesture    = None
        palm_center = (w // 2, h // 2)

        if self._latest_result and self._latest_result.hand_landmarks:
            for lm_list in self._latest_result.hand_landmarks:
                # 스켈레톤 그리기 (Tasks API)
                self._draw_utils.draw_landmarks(
                    frame,
                    lm_list,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    self._draw_styles.get_default_hand_landmarks_style(),
                    self._draw_styles.get_default_hand_connections_style(),
                )

                # 손바닥 중심 (WRIST 랜드마크)
                wrist = lm_list[WRIST]
                palm_center = (int(wrist.x * w), int(wrist.y * h))

                gesture = detect_gesture(lm_list)

        # 안정적 제스처 인식 (버퍼링)
        self._gesture_buffer.append(gesture)
        if len(self._gesture_buffer) > self.GESTURE_HOLD_FRAMES:
            self._gesture_buffer.pop(0)

        stable = self._stable_gesture()

        if stable:
            self.effect.activate(stable, palm_center)
        else:
            self.effect.deactivate()

        self._current_gesture = stable

        # 효과 렌더링
        self.effect.update_and_draw(frame)

        # 나루토 모드 표시 (우측 상단)
        self._draw_mode_indicator(frame)

        return frame

    # ── 내부 헬퍼 ──────────────────────────────────────

    def _stable_gesture(self):
        """버퍼 안에서 다수결로 안정된 제스처를 반환."""
        if not self._gesture_buffer:
            return None
        counts: dict[str, int] = {}
        for g in self._gesture_buffer:
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
        if not counts:
            return None
        best, cnt = max(counts.items(), key=lambda x: x[1])
        if cnt >= self.GESTURE_HOLD_FRAMES // 2:
            return best
        return None

    def _draw_mode_indicator(self, frame):
        h, w = frame.shape[:2]
        if self.enabled:
            label = "NARUTO MODE: ON"
            color = (0, 140, 255)
        else:
            label = "NARUTO MODE: OFF  [N]"
            color = (80, 80, 80)
        cv.putText(frame, label,
                   (w - 260, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55,
                   color, 2, cv.LINE_AA)

    @property
    def current_gesture(self):
        return self._current_gesture


# ───────────────────────────────────────────────
#  하단 패널 빌더 (video_recorder 에서 호출)
# ───────────────────────────────────────────────

# 제스처별 정보 (패널 표시용)
JUTSU_PANEL_INFO = [
    ("RASENGAN",     "✊ 주먹",        "🌀 RASENGAN",      (255, 160,  50)),  # 파랑
    ("KAGE_BUNSHIN", "✌  검지+중지",   "影 KAGE BUNSHIN",  (200, 200, 255)),  # 흰
    ("KATON",        "🖐 손 활짝",     "🔥 KATON",         ( 50,  80, 255)),  # 빨강
    ("RAIKIRI",      "☝ 검지만",      "⚡ RAIKIRI",        (100, 220, 255)),  # 노랑
    ("FUTON",        "🤙 엄지+새끼",   "🌪 FUTON",         (100, 220, 100)),  # 초록
    ("CHAKRA_FOCUS", "👆 엄지+검지",   "💜 CHAKRA",        (200,  80, 200)),  # 보라
]


def build_jutsu_panel(panel_w: int, panel_h: int,
                      active_jutsu: str | None,
                      naruto_enabled: bool,
                      is_recording: bool,
                      frame_count: int) -> np.ndarray:
    """
    하단 패널을 생성하여 numpy 배열로 반환한다.

    panel_w x panel_h 크기의 검은 배경에:
      - 좌측: 현재 활성 기술 정보 + 펄스 바
      - 우측: 6가지 제스처 카드 그리드
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # ── 배경 그라디언트 느낌 ─────────────────────────────
    panel[:] = (18, 18, 25)

    # ── 상단 구분선 ──────────────────────────────────────
    cv.line(panel, (0, 0), (panel_w, 0), (60, 60, 80), 2)

    LEFT_W  = panel_w // 3          # 좌측 영역 너비
    RIGHT_W = panel_w - LEFT_W      # 우측 영역 너비

    # ════════════════════════════════════════════════════
    #  좌측: 활성 기술 / 모드 표시
    # ════════════════════════════════════════════════════
    if not naruto_enabled:
        _draw_panel_text(panel, "NARUTO MODE: OFF", LEFT_W // 2, panel_h // 2,
                         (120, 120, 120), scale=0.7, center=True)
    elif active_jutsu is None:
        _draw_panel_text(panel, "NARUTO MODE: ON",  LEFT_W // 2, 28,
                         (0, 170, 255), scale=0.65, center=True)
        _draw_panel_text(panel, "제스처를 취해보세요",  LEFT_W // 2, panel_h // 2 - 10,
                         (160, 160, 160), scale=0.5, center=True)
        _draw_panel_text(panel, "손을 카메라에 보여주세요",  LEFT_W // 2, panel_h // 2 + 18,
                         (100, 100, 100), scale=0.42, center=True)
    else:
        info = JutsuEffect.JUTSU_INFO[active_jutsu]
        color = info["color"]
        bgr   = (color[2], color[1], color[0])

        # 기술 이름
        _draw_panel_text(panel, info["name"], LEFT_W // 2, 30,
                         bgr, scale=0.7, center=True, thickness=2)

        # 구분선
        cv.line(panel, (20, 45), (LEFT_W - 20, 45), bgr, 1)

        # 펄스 파워 바
        bar_x1, bar_y  = 20, 60
        bar_x2 = LEFT_W - 20
        bar_total = bar_x2 - bar_x1
        pulse = abs(math.sin(frame_count * 0.06))
        bar_fill = int(bar_total * pulse)
        cv.rectangle(panel, (bar_x1, bar_y),     (bar_x2,          bar_y + 12), (50, 50, 50), -1)
        cv.rectangle(panel, (bar_x1, bar_y),     (bar_x1 + bar_fill, bar_y + 12), bgr, -1)
        _draw_panel_text(panel, "CHAKRA", bar_x1, bar_y + 28,
                         (160, 160, 160), scale=0.38)

        # 미니 이펙트 아이콘 (원 + 회전점)
        cx, cy = LEFT_W // 2, (panel_h + 65) // 2
        angle = frame_count * 5.0
        cv.circle(panel, (cx, cy), 28, bgr, 2)
        for i in range(6):
            a  = math.radians(angle + i * 60)
            px = int(cx + 24 * math.cos(a))
            py = int(cy + 24 * math.sin(a))
            cv.circle(panel, (px, py), 4, bgr, -1)
        cv.circle(panel, (cx, cy), 10, bgr, -1)

    # 좌/우 구분 수직선
    cv.line(panel, (LEFT_W, 10), (LEFT_W, panel_h - 10), (60, 60, 80), 1)

    # ════════════════════════════════════════════════════
    #  우측: 6가지 제스처 카드 (2열 × 3행)
    # ════════════════════════════════════════════════════
    COLS, ROWS = 3, 2
    card_w = RIGHT_W // COLS
    card_h = panel_h // ROWS
    pad = 6

    for idx, (key, gesture_label, jutsu_label, color) in enumerate(JUTSU_PANEL_INFO):
        col = idx % COLS
        row = idx // COLS
        x0  = LEFT_W + col * card_w
        y0  = row * card_h

        is_active = (active_jutsu == key)

        # 카드 배경
        bg_color = (35, 35, 48) if not is_active else (
            min(255, color[0] // 4 + 20),
            min(255, color[1] // 4 + 20),
            min(255, color[2] // 4 + 20),
        )
        cv.rectangle(panel, (x0 + pad, y0 + pad),
                     (x0 + card_w - pad, y0 + card_h - pad), bg_color, -1)

        # 활성 카드 테두리
        border_color = color if is_active else (55, 55, 70)
        border_thick = 2 if is_active else 1
        cv.rectangle(panel, (x0 + pad, y0 + pad),
                     (x0 + card_w - pad, y0 + card_h - pad),
                     border_color, border_thick)

        text_color = color if is_active else (140, 140, 160)
        cx_card    = x0 + card_w // 2

        # 제스처 레이블
        _draw_panel_text(panel, gesture_label, cx_card, y0 + 22,
                         text_color, scale=0.42, center=True)
        # 기술 레이블
        _draw_panel_text(panel, jutsu_label,   cx_card, y0 + 42,
                         text_color if not is_active else color,
                         scale=0.44, center=True,
                         thickness=2 if is_active else 1)

    # ── 녹화 표시 (우측 하단) ────────────────────────────
    if is_recording:
        rx, ry = panel_w - 90, panel_h - 20
        cv.circle(panel, (rx, ry), 7, (0, 0, 255), -1)
        _draw_panel_text(panel, "REC", rx + 14, ry + 5, (0, 0, 255), scale=0.5)

    return panel


def _draw_panel_text(img, text, x, y, color, scale=0.5, thickness=1, center=False):
    """패널 텍스트 헬퍼."""
    font = cv.FONT_HERSHEY_SIMPLEX
    if center:
        (tw, th), _ = cv.getTextSize(text, font, scale, thickness)
        x = x - tw // 2
        y = y + th // 2
    cv.putText(img, text, (int(x), int(y)), font, scale, color, thickness, cv.LINE_AA)

