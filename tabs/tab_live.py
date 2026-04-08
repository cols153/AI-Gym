import time
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from pipeline.pose.pose_processor import PoseProcessor
from pipeline.pose.mediapipe_pose import MediaPipePose
from pipeline.runtime.live_state import LiveState


@st.cache_resource
def get_pose_pipe():
    return MediaPipePose()

def get_live_state() -> LiveState:
    if "live_state" not in st.session_state:
        st.session_state.live_state = LiveState()
    return st.session_state.live_state

def render_tab_live():
    live_state = get_live_state()

    with st.container(border=True):
        

        col1, col2 = st.columns(2, gap="large")
        rep_placeholder = col1.empty()
        status_placeholder = col2.empty()

        ctx = webrtc_streamer(
            key="pose-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: PoseProcessor(
                pipe=get_pose_pipe(),
                live_state=live_state,
                width=480,
                height=360,
                detect_every_n_frames=2,
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480},
                    "height": {"ideal": 360},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
        )

    # Initial render
    snap = live_state.snapshot()

    with rep_placeholder.container(border=True):
        st.metric("Repetitions", snap["reps"])

    with status_placeholder.container(border=True):
        st.metric("Status", snap["status"])

    # Live update loop while stream is active
    if ctx.state.playing:
        while ctx.state.playing:
            snap = live_state.snapshot()

            with rep_placeholder.container(border=True):
                st.metric("Repetitions", snap["reps"])

            with status_placeholder.container(border=True):
                st.metric("Status", snap["status"])

            time.sleep(0.5)