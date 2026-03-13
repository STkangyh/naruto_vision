import cv2 as cv
import numpy as np
from datetime import datetime
import os


class VideoRecorder:
    def __init__(self):
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 20.0
        self.output_folder = "recordings"
        
        # 녹화 파일 저장 폴더 생성
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def initialize_camera(self):
        """카메라 초기화"""
        self.cap = cv.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: 카메라를 열 수 없습니다.")
            return False
        
        # 카메라 해상도 설정
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)
        
        # 실제 설정된 값 가져오기
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        
        print(f"카메라 초기화 완료: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        return True
    
    def start_recording(self):
        """녹화 시작"""
        if self.is_recording:
            return
        
        # 파일명 생성 (타임스탬프 사용)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_folder, f"recording_{timestamp}.avi")
        
        # VideoWriter 초기화
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.writer = cv.VideoWriter(
            filename,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.writer.isOpened():
            print("Error: VideoWriter를 초기화할 수 없습니다.")
            return
        
        self.is_recording = True
        print(f"녹화 시작: {filename}")
    
    def stop_recording(self):
        """녹화 중지"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        print("녹화 중지")
    
    def draw_recording_indicator(self, frame):
        """녹화 중 표시 (빨간 원과 REC 텍스트)"""
        # 빨간 원 그리기 (좌측 상단)
        cv.circle(frame, (30, 30), 15, (0, 0, 255), -1)
        
        # REC 텍스트 추가
        cv.putText(
            frame,
            "REC",
            (55, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # 녹화 시간 표시 (옵션)
        cv.putText(
            frame,
            "Recording...",
            (55, 65),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
    
    def draw_preview_indicator(self, frame):
        """Preview 모드 표시"""
        cv.putText(
            frame,
            "PREVIEW",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # 사용 방법 안내
        cv.putText(
            frame,
            "SPACE: Start/Stop Recording | ESC: Exit",
            (10, self.frame_height - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def run(self):
        """메인 루프 실행"""
        if not self.initialize_camera():
            return
        
        print("\n=== Video Recorder ===")
        print("SPACE: 녹화 시작/중지")
        print("ESC: 프로그램 종료")
        print("=====================\n")
        print("카메라 창이 열립니다. 카메라 창을 클릭하여 활성화하세요.")
        
        # 윈도우 생성 및 위치 설정
        cv.namedWindow('Video Recorder', cv.WINDOW_NORMAL)
        cv.resizeWindow('Video Recorder', self.frame_width, self.frame_height)
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: 프레임을 읽을 수 없습니다.")
                    break
                
                # 프레임 처리
                display_frame = frame.copy()
                
                if self.is_recording:
                    # Record 모드
                    self.draw_recording_indicator(display_frame)
                    # 프레임 저장
                    self.writer.write(frame)
                else:
                    # Preview 모드
                    self.draw_preview_indicator(display_frame)
                
                # 화면에 표시 - 실시간 카메라 영상
                cv.imshow('Video Recorder', display_frame)
                
                # 키 입력 처리
                key = cv.waitKey(1) & 0xFF
                
                if key == 27:  # ESC 키
                    print("프로그램 종료")
                    break
                elif key == 32:  # SPACE 키
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
        
        finally:
            # 정리
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.is_recording:
            self.stop_recording()
        
        if self.cap is not None:
            self.cap.release()
        
        cv.destroyAllWindows()
        print("리소스 정리 완료")


def main():
    recorder = VideoRecorder()
    recorder.run()


if __name__ == "__main__":
    main()
