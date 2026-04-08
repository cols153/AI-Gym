import time
import numpy as np
import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase


class PoseProcessor(VideoProcessorBase):
    
    def __init__(self, pipe, live_state, width=480, height=360, detect_every_n_frames=2):
        self.pipe = pipe
        self.live_state = live_state
        self.width = width
        self.height = height
        self.detect_every_n_frames = detect_every_n_frames

        self.start_time = time.perf_counter()
        self.frame_idx = 0
        self.last_result = None

    def _analyze_result(self, result) -> dict:
        """
        TEMP fake logic for now.
        Replace later with real pose-based logic.
        """
        is_correct = self.frame_idx % 40 < 20
        rep_completed = self.frame_idx % 60 == 0

        return {
            "is_correct": is_correct,
            "rep_completed": rep_completed,
        }

    def _update_live_state(self, analysis: dict):
        if analysis["is_correct"]:
            self.live_state.set_status("Correct")
        else:
            self.live_state.set_status("Incorrect")

        if analysis["rep_completed"]:
            self.live_state.increment_rep()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        image_bgr = cv2.resize(image_bgr, (self.width, self.height))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self.frame_idx += 1

        if self.frame_idx % self.detect_every_n_frames == 0:
            timestamp_ms = int((time.perf_counter() - self.start_time) * 1000)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb,
            )

            self.last_result = self.pipe.detect(mp_image, timestamp_ms)

            if self.last_result is not None:
                analysis = self._analyze_result(self.last_result)
                self._update_live_state(analysis)

        output_bgr = image_bgr

        if self.last_result is not None:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb,
            )
            annotated_rgb = self.pipe.draw(mp_image, self.last_result)
            output_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        output_bgr = np.asarray(output_bgr, dtype=np.uint8)

        return av.VideoFrame.from_ndarray(output_bgr, format="bgr24")