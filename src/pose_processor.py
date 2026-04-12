import time
import numpy as np
import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase
from src.Pipeline import Pipeline


class PoseProcessor(VideoProcessorBase):
    def __init__(self, pose, state, mode=2):
        self.pose = pose
        self.state = state
        self.mode = mode

        # Lazy init
        self.pipe = None

    def _ensure_pipeline(self):
        if self.pipe is None:
            self.pipe = Pipeline(mode=self.mode, state=self.state)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Ensure pipeline is loaded only when needed
        self._ensure_pipeline()

        image_bgr = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        timestamp_ms = int(time.time() * 1000)
        result = self.pose.detect(mp_image, timestamp_ms)

        # Pose detected
        if result is not None:
            # Add frame detection to pipeline
            self.pipe.submit(result) # type: ignore

            # Draw on image
            annotated_rgb = self.pose.draw(mp_image, result)
            output_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        else:
            output_bgr = image_bgr

        return av.VideoFrame.from_ndarray(output_bgr.astype(np.uint8), format="bgr24")