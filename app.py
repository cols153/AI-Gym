import time
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.pose_processor import PoseProcessor
from src.mediapipe_pose import MediaPipePose
from src.State import State

st.set_page_config(
    page_title="Push-up trainer",
    page_icon="data/coach.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def get_pose_pipe():
    return MediaPipePose()

def get_state() -> State:
    if "live_state" not in st.session_state:
        st.session_state.live_state = State()
    return st.session_state.live_state

def coach_speak(text):
    full_text = ""

    for word in text.split():
        full_text += word + " "
        coach_text.markdown(full_text)
        time.sleep(0.05)


left, center, right = st.columns([1, 2, 1])

with center:

    col1, col2 =  st.columns([1,3],vertical_alignment="center")

    with col1:
        st.image("data/coach.png", width=200)

    with col2:    
        coach_text = st.empty()
        coach_speak("Hey, I am your coach. Lets get started with the push-up training!")

    state = get_state()

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

    with st.expander("Session history", expanded=False):
        history_box = st.empty()

    interval = 2.0 if ctx.state.playing else None

    @st.fragment(run_every=interval)
    def live_metrics():
        snap = state.snapshot()

        last = snap["status"]

        correct_delta = "+1" if last == "correct" else None
        incorrect_delta = "+1" if last == "incorrect" else None

        correct_box.metric("Correct", snap["correct_reps"], correct_delta)
        incorrect_box.metric("Incorrect", snap["incorrect_reps"], incorrect_delta)

        coach_speak(snap["feedback"])

        with history_box.container():
            for msg in snap["history"]:
                with st.chat_message("Coach"):
                    st.write(
                        f"Rep {msg['rep']}: {msg['label']} "
                        f"({msg['confidence']:.2f})"
            )


with right:
    with st.container(border=True):
        st.markdown("<p style='text-align: center;'>Scoreboard</p>", unsafe_allow_html=True)

        m1, m2 = st.columns(2)

        with m1:
            correct_box = st.empty()

        with m2:
            incorrect_box = st.empty()

        correct_box.metric("Correct", 0)
        incorrect_box.metric("Incorrect", 0)

    if st.button("Clear session", use_container_width=True):
        st.session_state.live_state = State()
        st.rerun()

    live_metrics()
