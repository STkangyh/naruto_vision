import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math
import os
from PIL import Image, ImageDraw, ImageFont

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


def _palm_facing_up(lm):
    """
    손바닥이 위를 향하는지 판별.
    손목(WRIST)의 z가 중지 MCP(MIDDLE_MCP)의 z보다 크면
    손등이 카메라 쪽 → 손바닥이 위(천장)를 향하는 상태.
    """
    return lm[WRIST].z > lm[MIDDLE_MCP].z


def _only_index_middle_up(lm):
    """검지·중지는 펴고, 약지·새끼는 접힌 상태 (엄지는 무관)."""
    index_up  = _tip_above_mcp(lm, INDEX_TIP,  INDEX_MCP)
    middle_up = _tip_above_mcp(lm, MIDDLE_TIP, MIDDLE_MCP)
    ring_up   = _tip_above_mcp(lm, RING_TIP,   RING_MCP)
    pinky_up  = _tip_above_mcp(lm, PINKY_TIP,  PINKY_MCP)
    return index_up and middle_up and not ring_up and not pinky_up


def detect_katon_two_hands(lm_list_all: list) -> bool:
    """
    KATON 제스처: 두 손 모두 검지+중지+엄지가 위로 핀 상태.
    lm_list_all: 감지된 모든 손의 랜드마크 리스트.
    """
    if len(lm_list_all) < 2:
        return False
    return all(_only_index_middle_up(lm) for lm in lm_list_all)


def count_fingers(lm):
    """펴진 손가락 수 반환 (엄지 제외 4개 + 엄지 1개)."""
    count = 0
    if lm[THUMB_TIP].x < lm[THUMB_MCP].x:
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
    단일 손 랜드마크에서 나루토 기술 제스처를 판별한다.
    KATON은 양손 제스처이므로 detect_katon_two_hands() 로 별도 처리.

    반환값:
        str | None  – 감지된 기술 이름 또는 None
    """
    thumb_up   = lm[THUMB_TIP].x  < lm[THUMB_MCP].x
    index_up   = _tip_above_mcp(lm, INDEX_TIP,  INDEX_MCP)
    middle_up  = _tip_above_mcp(lm, MIDDLE_TIP, MIDDLE_MCP)
    ring_up    = _tip_above_mcp(lm, RING_TIP,   RING_MCP)
    pinky_up   = _tip_above_mcp(lm, PINKY_TIP,  PINKY_MCP)

    n_up = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    # ── 라센간: 손 활짝 핌 + 손바닥이 위를 향함 ─────────
    if n_up == 5 and _palm_facing_up(lm):
        return "RASENGAN"

    # ── 차크라 집중: 엄지+검지만 핀 (총기 모양) ──────────
    if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "CHAKRA_FOCUS"

    # ── 카게부신: 검지+중지 (V사인), 나머지 접힘 ──────────
    if not thumb_up and index_up and middle_up and not ring_up and not pinky_up:
        return "KAGE_BUNSHIN"

    # ── 뇨이보: 검지만 핌 ─────────────────────────────────
    if not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "RAIKIRI"

    # ── 풍둔: 엄지+새끼 (shaka) ──────────────────────────
    if thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
        return "FUTON"

    return None


# ───────────────────────────────────────────────
#  파티클 시스템 (효과 연출용)
# ───────────────────────────────────────────────

class Particle:
    def __init__(self, x, y, color, speed=3, size=4, lifetime=30,
                 vx=None, vy=None, gravity=0.15, shrink=0.1, kind="normal"):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.size = float(size)
        self.lifetime = lifetime
        self.age = 0
        self.gravity = gravity
        self.shrink = shrink
        self.kind = kind          # "normal" | "fire" | "ember" | "smoke"
        if vx is not None and vy is not None:
            self.vx = float(vx)
            self.vy = float(vy)
        else:
            angle = np.random.uniform(0, 2 * math.pi)
            spd   = np.random.uniform(speed * 0.5, speed * 1.5)
            self.vx = math.cos(angle) * spd
            self.vy = math.sin(angle) * spd

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        self.vy += self.gravity
        self.size = max(0.5, self.size - self.shrink)
        # 불꽃은 수평으로 서서히 퍼짐
        if self.kind in ("fire", "ember"):
            self.vx *= 0.96

    @property
    def alive(self):
        return self.age < self.lifetime and self.size > 0.5

    def draw(self, frame):
        alpha = 1.0 - self.age / self.lifetime
        r, g, b = self.color
        ix, iy = int(self.x), int(self.y)
        sz = max(1, int(self.size))
        h, w = frame.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return

        if self.kind == "smoke":
            # 연기: 반투명 회색 원
            overlay = frame.copy()
            cv.circle(overlay, (ix, iy), sz,
                      (int(180 * alpha), int(180 * alpha), int(180 * alpha)), -1)
            cv.addWeighted(overlay, 0.18 * alpha, frame, 1 - 0.18 * alpha, 0, frame)
        else:
            cv.circle(frame, (ix, iy), sz,
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
        # KATON 전용 상태
        self._katon_ball_x  = 0.0
        self._katon_ball_y  = 0.0
        self._katon_phase   = "charge"  # "charge" → "expand" → "smoke" → "clear"
        self._katon_radius  = 0.0
        self._katon_vx      = 0.0
        self._katon_vy      = 0.0

    # ── 외부 인터페이스 ────────────────────────────────

    def activate(self, jutsu_name: str, palm_center: tuple):
        """기술 활성화 (같은 기술이면 유지, 다른 기술이면 교체)."""
        if jutsu_name != self.active_jutsu:
            self.active_jutsu = jutsu_name
            self.frame_count  = 0
            self.particles.clear()
            self._spin_angle  = 0.0
            # KATON 초기화
            if jutsu_name == "KATON":
                cx, cy = palm_center
                self._katon_ball_x = float(cx)
                self._katon_ball_y = float(cy)
                self._katon_phase  = "charge"
                self._katon_radius = 0.0
                self._katon_vx     = 0.0
                self._katon_vy     = 0.0
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

        # KATON은 파티클/효과 좌표를 화구 위치 기준으로 처리
        spawn_cx, spawn_cy = cx, cy
        if self.active_jutsu == "KATON":
            spawn_cx = int(self._katon_ball_x)
            spawn_cy = int(self._katon_ball_y)

        # smoke/clear 단계는 파티클 새로 생성 안 함 (기존 파티클만 소멸)
        if self.active_jutsu == "KATON" and self._katon_phase in ("smoke", "clear"):
            pass
        else:
            self._spawn_particles(spawn_cx, spawn_cy, color)

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
            # 코어 크기에 비례해서 파티클 속도·크기·수 증가
            core_r = min(18 + self.frame_count * 0.6, 120)
            count  = int(8 + core_r * 0.12)          # 최대 ~22개
            spd    = 4 + core_r * 0.05               # 최대 ~10
            sz     = 5 + core_r * 0.06               # 최대 ~13
            for _ in range(count):
                self.particles.append(
                    Particle(cx, cy, color,
                             speed=spd, size=sz, lifetime=28))
        elif jutsu == "KATON":
            # charge 단계: 손 주변에 불꽃이 안쪽으로 모임
            if self._katon_phase == "charge":
                bx, by = int(self._katon_ball_x), int(self._katon_ball_y)
                for _ in range(8):
                    ang = np.random.uniform(0, 2 * math.pi)
                    r   = np.random.uniform(self._katon_radius * 0.8,
                                            self._katon_radius * 1.4 + 10)
                    px  = bx + math.cos(ang) * r
                    py  = by + math.sin(ang) * r
                    dvx = (bx - px) * 0.12 + np.random.uniform(-1.5, 1.5)
                    dvy = (by - py) * 0.12 + np.random.uniform(-1.5, 1.5)
                    fc  = (np.random.randint(220, 255),
                           np.random.randint(80, 200), 0)
                    self.particles.append(
                        Particle(px, py, fc,
                                 size=np.random.uniform(4, 10),
                                 lifetime=14, vx=dvx, vy=dvy,
                                 gravity=-0.04, shrink=0.25, kind="fire"))
            # expand 단계: 화구 외곽에서 불꽃이 사방으로 터짐
            elif self._katon_phase == "expand":
                bx, by = int(self._katon_ball_x), int(self._katon_ball_y)
                for _ in range(14):
                    ang = np.random.uniform(0, 2 * math.pi)
                    spd = np.random.uniform(5, 18)
                    fc  = (np.random.randint(210, 255),
                           np.random.randint(60, 180), 0)
                    self.particles.append(
                        Particle(bx, by, fc,
                                 size=np.random.uniform(8, 22),
                                 lifetime=np.random.randint(20, 40),
                                 vx=math.cos(ang) * spd,
                                 vy=math.sin(ang) * spd,
                                 gravity=0.06, shrink=0.2, kind="fire"))
                # 불씨
                for _ in range(8):
                    ang = np.random.uniform(0, 2 * math.pi)
                    spd = np.random.uniform(3, 10)
                    ec  = (np.random.randint(200, 255),
                           np.random.randint(120, 220), 0)
                    self.particles.append(
                        Particle(bx, by, ec,
                                 size=np.random.uniform(2, 5),
                                 lifetime=np.random.randint(30, 60),
                                 vx=math.cos(ang) * spd,
                                 vy=math.sin(ang) * spd,
                                 gravity=0.15, shrink=0.04, kind="ember"))
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
            self._draw_rasengan(frame, cx, cy, bgr, angle, t)

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
    def _draw_rasengan(self, frame, cx, cy, bgr, angle, t):
        # 유지 시간에 따라 코어 반지름 성장 (최대 120px)
        core_r  = int(min(18 + t * 0.6, 120))
        # 궤도 링도 코어에 맞춰 함께 커짐
        orbit_r = core_r + 27
        ring_max = core_r + 42

        overlay = frame.copy()
        # 바깥 나선 고리 (코어 크기에 맞춰 동적으로)
        for r in range(ring_max, core_r, -10):
            cv.circle(overlay, (cx, cy), r, bgr, 2)
        cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 회전하는 궤도 점
        for i in range(12):
            a  = math.radians(angle + i * 30)
            px = int(cx + orbit_r * math.cos(a))
            py = int(cy + orbit_r * math.sin(a))
            dot_r = max(3, int(4 + core_r * 0.04))
            cv.circle(frame, (px, py), dot_r, bgr, -1)

        # 글로우 (코어가 커질수록 강해짐)
        glow_alpha = min(0.45, 0.1 + t * 0.003)
        glow_overlay = frame.copy()
        cv.circle(glow_overlay, (cx, cy), core_r + 20, bgr, -1)
        cv.addWeighted(glow_overlay, glow_alpha, frame, 1 - glow_alpha, 0, frame)

        # 중심 코어
        cv.circle(frame, (cx, cy), core_r, bgr, -1)
        cv.circle(frame, (cx, cy), core_r + 4, (255, 255, 255), 2)
        # 코어 흰 중심점 (항상 밝게)
        cv.circle(frame, (cx, cy), max(6, core_r // 4), (255, 255, 255), -1)

    # ─── 화둔 ────────────────────────────────────────────
    def _draw_katon(self, frame, cx, cy, t):
        """화둔·호화구: charge → expand → smoke → clear 4단계."""
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)          # 화면 대각선 길이

        # ── 단계 전이 타이머 (프레임 수 기준) ─────────────
        # charge : 0~24  (불덩이 커지기)
        # expand : 25~54 (화면 전체로 확산)
        # smoke  : 55~84 (검은 연기로 뿌옇게)
        # clear  : 85~   (연기 걷힘 → 자동 deactivate)

        CHARGE_END = 24
        EXPAND_END = 54
        SMOKE_END  = 84

        if self._katon_phase == "charge":
            self._katon_ball_x = float(cx)
            self._katon_ball_y = float(cy)
            self._katon_radius = min(t * 3.0, 60.0)
            if t >= CHARGE_END:
                self._katon_phase  = "expand"
                self._katon_radius = 60.0

        elif self._katon_phase == "expand":
            prog = (t - CHARGE_END) / (EXPAND_END - CHARGE_END)  # 0→1
            prog = min(prog, 1.0)
            # 화구 반지름: 60 → 화면 대각선의 0.8배
            self._katon_radius = 60.0 + prog * diag * 0.85
            if t >= EXPAND_END:
                self._katon_phase = "smoke"

        elif self._katon_phase == "smoke":
            if t >= SMOKE_END:
                self._katon_phase = "clear"

        elif self._katon_phase == "clear":
            if t >= SMOKE_END + 35:
                # 이펙트 종료
                self.deactivate()
                return

        # ── 렌더링 ──────────────────────────────────────
        bx = int(self._katon_ball_x)
        by = int(self._katon_ball_y)
        br = int(self._katon_radius)

        if self._katon_phase == "charge":
            flicker = int(6 * math.sin(t * 1.2))
            # 글로우
            overlay = frame.copy()
            for gr, gc in [(br + flicker + 30, (0, 20, 180)),
                           (br + flicker + 16, (0, 60, 230)),
                           (br + flicker +  5, (0, 110, 255))]:
                if gr > 0:
                    cv.circle(overlay, (bx, by), gr, gc, -1)
            cv.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)
            # 화구 본체
            for fr_, fc_ in [(br + flicker,      (0,  55, 255)),
                             (br + flicker - 10,  (0, 130, 255)),
                             (br + flicker - 20,  (0, 200, 255)),
                             (max(br // 3, 8),    (210, 245, 255))]:
                if fr_ > 0:
                    cv.circle(frame, (bx, by), fr_, fc_, -1)
            # 표면 불꽃
            np.random.seed(t % 60)
            for _ in range(10):
                ang  = np.random.uniform(0, 2 * math.pi)
                dist = np.random.uniform(br * 0.5, br * 1.0 + flicker)
                tx = bx + int(math.cos(ang) * dist)
                ty = by + int(math.sin(ang) * dist)
                cv.circle(frame, (tx, ty), np.random.randint(4, 11),
                          (np.random.randint(0, 60),
                           np.random.randint(100, 210), 255), -1)

        elif self._katon_phase == "expand":
            prog = (t - CHARGE_END) / (EXPAND_END - CHARGE_END)
            prog = min(prog, 1.0)
            # 화면 전체를 덮는 불덩이
            # 바깥부터: 주황 → 안쪽: 흰 노랑
            # alpha는 prog에 따라 강해짐
            fire_alpha = min(0.92, 0.3 + prog * 0.65)
            overlay = frame.copy()
            overlay[:] = (0, 50, 255)          # 주황 (BGR)
            cv.addWeighted(overlay, fire_alpha, frame, 1 - fire_alpha, 0, frame)

            # 코어 원 (밝은 노랑)
            core_r = max(int(br * (1.0 - prog * 0.6)), 10)
            core_col = (int(200 * (1 - prog)), int(235 * (1 - prog * 0.3)), 255)
            cv.circle(frame, (bx, by), core_r, core_col, -1)

            # 화면 가장자리 글로우
            for margin in [0, 6, 14]:
                pts = np.array([[margin, margin],
                                [w - margin, margin],
                                [w - margin, h - margin],
                                [margin, h - margin]], np.int32)
                intensity = int(200 * (1 - margin / 20.0) * prog)
                cv.polylines(frame, [pts], True,
                             (0, intensity // 3, intensity), 3)

        elif self._katon_phase == "smoke":
            prog = (t - EXPAND_END) / (SMOKE_END - EXPAND_END)
            prog = min(prog, 1.0)
            # 검은 연기로 점점 뿌옇게
            smoke_alpha = 0.15 + prog * 0.72
            overlay = frame.copy()
            overlay[:] = (30, 30, 30)
            cv.addWeighted(overlay, smoke_alpha, frame, 1 - smoke_alpha, 0, frame)
            # 흰 연기 질감 노이즈
            noise_alpha = prog * 0.25
            noise = np.random.randint(160, 220, frame.shape, dtype=np.uint8)
            cv.addWeighted(noise, noise_alpha, frame, 1 - noise_alpha, 0, frame)

        elif self._katon_phase == "clear":
            prog = (t - SMOKE_END) / 35.0
            prog = min(prog, 1.0)
            # 연기가 걷히며 점점 투명해짐
            smoke_alpha = max(0.0, 0.87 - prog * 0.87)
            if smoke_alpha > 0.01:
                overlay = frame.copy()
                overlay[:] = (30, 30, 30)
                cv.addWeighted(overlay, smoke_alpha, frame, 1 - smoke_alpha, 0, frame)
                noise_alpha = smoke_alpha * 0.25
                noise = np.random.randint(160, 220, frame.shape, dtype=np.uint8)
                cv.addWeighted(noise, noise_alpha, frame, 1 - noise_alpha, 0, frame)

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
        h, w = frame.shape[:2]

        # ── 3초(90프레임) 이상 유지 시 분신 2개 등장 ─────
        BUNSHIN_THRESHOLD = 90  # frames
        if t >= BUNSHIN_THRESHOLD:
            # 등장 진행도 0→1 (30프레임에 걸쳐 서서히 나타남)
            appear = min((t - BUNSHIN_THRESHOLD) / 30.0, 1.0)

            # 분신 오프셋: 좌우로 벌어짐
            spread = int(appear * 90)
            offsets = [-spread, spread]   # 왼쪽 분신, 오른쪽 분신

            for off_x in offsets:
                # 프레임을 옆으로 밀어서 반투명하게 합성
                M = np.float32([[1, 0, off_x], [0, 1, 0]])
                ghost = cv.warpAffine(frame, M, (w, h))

                # 연기 마스크: 분신 중심부를 흰색으로 밝힘 (등장 연출)
                smoke_overlay = ghost.copy()
                bx = cx + off_x
                smoke_r = int(60 + 20 * math.sin(t * 0.15))
                cv.circle(smoke_overlay, (bx, cy), smoke_r, (220, 220, 220), -1)
                cv.addWeighted(smoke_overlay, 0.25 * appear, ghost, 1 - 0.25 * appear, 0, ghost)

                # 분신을 메인 프레임에 반투명 합성
                bunshin_alpha = 0.45 * appear
                cv.addWeighted(ghost, bunshin_alpha, frame, 1 - bunshin_alpha, 0, frame)

                # 분신 윤곽선 (파란 테두리 원)
                if bx >= 0 and bx < w:
                    cv.circle(frame, (bx, cy), int(30 * appear), bgr, 2)

        # ── 연기 효과 (항상) ─────────────────────────────
        overlay = frame.copy()
        for i in range(3):
            offset_x = int(40 * math.cos(math.radians(t * 3 + i * 120)))
            offset_y = int(20 * math.sin(math.radians(t * 3 + i * 120)))
            r = 30 + int(5 * math.sin(t * 0.2 + i))
            cv.circle(overlay, (cx + offset_x, cy + offset_y), r,
                      (220, 220, 220), -1)
        cv.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # ── 중심 파란 도장 원 ────────────────────────────
        cv.circle(frame, (cx, cy), 16, bgr, -1)
        cv.circle(frame, (cx, cy), 20, (255, 255, 255), 2)

        # ── 3초 대기 중: 카운트다운 표시 ────────────────
        if t < BUNSHIN_THRESHOLD:
            remain = (BUNSHIN_THRESHOLD - t) / 30.0  # 남은 초
            bar_w  = 80
            bar_h  = 6
            bx0    = cx - bar_w // 2
            by0    = cy + 30
            filled = int(bar_w * (1.0 - remain / 3.0))
            cv.rectangle(frame, (bx0, by0), (bx0 + bar_w, by0 + bar_h),
                         (80, 80, 80), -1)
            cv.rectangle(frame, (bx0, by0), (bx0 + filled, by0 + bar_h),
                         bgr, -1)

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
            num_hands=2,
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
            all_lm = self._latest_result.hand_landmarks

            # 모든 손 스켈레톤 먼저 그리기
            for lm_list in all_lm:
                self._draw_utils.draw_landmarks(
                    frame,
                    lm_list,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    self._draw_styles.get_default_hand_landmarks_style(),
                    self._draw_styles.get_default_hand_connections_style(),
                )

            # ── KATON: 양손 모두 검지+중지만 위로 핀 상태 ──
            if detect_katon_two_hands(all_lm):
                gesture = "KATON"
                # 효과 중심: 두 손 중지 끝(MIDDLE_TIP)의 중간점
                mx = sum(lm[MIDDLE_TIP].x for lm in all_lm) / len(all_lm)
                my = sum(lm[MIDDLE_TIP].y for lm in all_lm) / len(all_lm)
                palm_center = (int(mx * w), int(my * h))
            else:
                # ── 단일 손 제스처 (첫 번째로 인식된 손) ───
                for lm_list in all_lm:
                    wrist = lm_list[WRIST]
                    palm_center = (int(wrist.x * w), int(wrist.y * h))
                    g = detect_gesture(lm_list)
                    if g is not None:
                        gesture = g
                        # KATON이면 효과 중심을 중지 끝(MIDDLE_TIP)으로
                        if g == "KATON":
                            palm_center = (
                                int(lm_list[MIDDLE_TIP].x * w),
                                int(lm_list[MIDDLE_TIP].y * h),
                            )
                        break

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
    ("RASENGAN",     "손 활짝 + 손바닥 위",   "RASENGAN",     (255, 160,  50)),
    ("KAGE_BUNSHIN", "검지 + 중지 (V)",        "KAGE BUNSHIN", (200, 200, 255)),
    ("KATON",        "[양손] 검지+중지 위로",   "KATON",        ( 50,  80, 255)),
    ("RAIKIRI",      "검지만",                 "RAIKIRI",      (100, 220, 255)),
    ("FUTON",        "엄지 + 새끼",            "FUTON",        (100, 220, 100)),
    ("CHAKRA_FOCUS", "엄지 + 검지",            "CHAKRA FOCUS", (200,  80, 200)),
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
        _draw_panel_text(panel, "NARUTO MODE: OFF", LEFT_W // 2, panel_h // 2 - 10,
                         (120, 120, 120), scale=0.7, center=True)
    elif active_jutsu is None:
        _draw_panel_text(panel, "NARUTO MODE: ON",  LEFT_W // 2, 14,
                         (0, 170, 255), scale=0.65, center=True)
        _draw_panel_text(panel, "제스처를 취해보세요",  LEFT_W // 2, panel_h // 2 - 18,
                         (160, 160, 160), scale=0.5, center=True)
        _draw_panel_text(panel, "손을 카메라에 보여주세요",  LEFT_W // 2, panel_h // 2 + 6,
                         (100, 100, 100), scale=0.42, center=True)
        _draw_panel_text(panel, "* 오른손 기준",  LEFT_W // 2, panel_h - 26,
                         (80, 130, 80), scale=0.40, center=True)
    else:
        info = JutsuEffect.JUTSU_INFO[active_jutsu]
        color = info["color"]
        bgr   = (color[2], color[1], color[0])

        # 기술 이름
        _draw_panel_text(panel, info["name"], LEFT_W // 2, 12,
                         bgr, scale=0.7, center=True, thickness=2)

        # 구분선
        cv.line(panel, (20, 40), (LEFT_W - 20, 40), bgr, 1)

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

    # 우측 상단: 오른손 기준 안내
    _draw_panel_text(panel, "* 오른손 기준",
                     LEFT_W + RIGHT_W - 8, 6,
                     (80, 130, 80), scale=0.38, center=False)

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
        _draw_panel_text(panel, gesture_label, cx_card, y0 + 10,
                         text_color, scale=0.42, center=True)
        # 기술 레이블
        _draw_panel_text(panel, jutsu_label,   cx_card, y0 + 32,
                         text_color if not is_active else color,
                         scale=0.44, center=True,
                         thickness=2 if is_active else 1)

    # ── 녹화 표시 (우측 하단) ────────────────────────────
    if is_recording:
        rx, ry = panel_w - 90, panel_h - 20
        cv.circle(panel, (rx, ry), 7, (0, 0, 255), -1)
        _draw_panel_text(panel, "REC", rx + 14, ry + 5, (0, 0, 255), scale=0.5)

    return panel


# ───────────────────────────────────────────────
#  PIL 기반 유니코드/이모지 텍스트 렌더러
# ───────────────────────────────────────────────

# 폰트 경로 (macOS 기본 경로)
_FONT_KO   = "/System/Library/Fonts/AppleSDGothicNeo.ttc"          # 한글+영문
_FONT_EMOJI = "/System/Library/Fonts/Apple Color Emoji.ttc"        # 이모지


def _load_font(size: int, emoji: bool = False):
    """PIL 폰트 로드 (실패 시 기본 폰트 반환)."""
    try:
        path = _FONT_EMOJI if emoji else _FONT_KO
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _pil_put_text(img_bgr: np.ndarray, text: str,
                  x: int, y: int,
                  color_bgr: tuple,
                  font_size: int = 14,
                  center: bool = False):
    """
    PIL로 유니코드/이모지 문자열을 img_bgr(numpy BGR) 위에 그린다.
    이모지 문자가 포함된 경우 이모지 폰트로 먼저 그리고,
    나머지 한글/영문은 한글 폰트로 겹쳐 그린다.

    x, y  : 텍스트 좌측 상단 (center=True 이면 가로 중앙 정렬 기준 x)
    """
    # BGR → RGB PIL Image
    pil_img = Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    font_ko    = _load_font(font_size, emoji=False)
    font_emoji = _load_font(font_size, emoji=True)

    # 텍스트 크기 측정 (한글 폰트 기준)
    bbox = font_ko.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw_x = (x - tw // 2) if center else x
    draw_y = y

    # PIL color: RGB tuple
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]

    # 이모지 문자는 이모지 폰트로, 나머지는 한글 폰트로
    # → 간단하게: 한글 폰트로 전체 그리기 (이모지는 ?)
    #             + 이모지 문자만 이모지 폰트로 덮어 그리기
    draw.text((draw_x, draw_y), text, font=font_ko, fill=(r, g, b))

    # 이모지가 있으면 위치를 직접 계산해서 이모지 폰트로 덮어쓰기
    cur_x = draw_x
    for ch in text:
        cp = ord(ch)
        is_emoji = (
            0x1F000 <= cp <= 0x1FFFF or   # 이모지 메인 블록
            0x2600  <= cp <= 0x27BF  or   # 기타 기호
            0x2300  <= cp <= 0x23FF  or   # 기술 기호
            cp in (0x261D, 0x270C, 0x270B, 0x1F44A, 0x1F91D)
        )
        ch_bbox = font_ko.getbbox(ch)
        ch_w = ch_bbox[2] - ch_bbox[0]
        if is_emoji:
            draw.text((cur_x, draw_y), ch, font=font_emoji, fill=(r, g, b))
        cur_x += ch_w

    # RGB PIL → BGR numpy
    result = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    np.copyto(img_bgr, result)


def _draw_panel_text(img, text, x, y, color, scale=0.5, thickness=1, center=False):
    """
    패널 텍스트 헬퍼.
    한글/이모지가 포함된 경우 PIL 렌더러로 위임,
    ASCII 전용이면 OpenCV putText 사용.
    """
    # 한글 또는 이모지 코드포인트가 있으면 PIL 사용
    needs_pil = any(ord(c) > 127 for c in text)
    if needs_pil:
        # scale 0.5 → 약 14px, 0.7 → 약 18px 로 매핑
        font_size = max(10, int(scale * 28))
        _pil_put_text(img, text, int(x), int(y), color,
                      font_size=font_size, center=center)
    else:
        font = cv.FONT_HERSHEY_SIMPLEX
        if center:
            (tw, th), _ = cv.getTextSize(text, font, scale, thickness)
            x = x - tw // 2
            y = y + th // 2
        cv.putText(img, text, (int(x), int(y)), font, scale,
                   color, thickness, cv.LINE_AA)
