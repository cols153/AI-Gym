import time
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.pose.pose_processor import PoseProcessor
from src.pose.mediapipe_pose import MediaPipePose
from src.runtime.State import State


@st.cache_resource
def get_pose_pipe():
    return MediaPipePose()

def get_state() -> State:
    if "live_state" not in st.session_state:
        st.session_state.live_state = State()
    return st.session_state.live_state

def render_tab_live():
    state = get_state()

    with st.container(border=True):
        
        col1, col2 = st.columns(2, gap="large")
        rep_placeholder = col1.empty()
        status_placeholder = col2.empty()

        ctx = webrtc_streamer(
            key="pose-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: PoseProcessor(
                pose=get_pose_pipe(),
                state=state,
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


    from streamlit_autorefresh import st_autorefresh

    if ctx.state.playing:
        st_autorefresh(interval=500, key="live_refresh")

    snap = state.snapshot()

    with rep_placeholder.container(border=True):
        st.metric("Repetitions", snap["reps"])

    with status_placeholder.container(border=True):
        st.metric("Status", snap["status"] or "-")