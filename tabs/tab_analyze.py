import os
import tempfile
from pathlib import Path
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pipeline.pipeline import run_video_pipeline

from pipeline.pose.pose_processor import PoseProcessor
from pipeline.pose.mediapipe_pose import MediaPipePose


def render_tab_analyze():
    st.session_state.setdefault("analyze_view", "input")
    st.session_state.setdefault("analyze_output_video", None)
    st.session_state.setdefault("analyze_results", None)

    if st.session_state.analyze_view == "results":
        render_results_view()
    else:
        render_input_view()

def render_input_view():
    video_path = None

    with st.expander("Expand for instructions"):
        st.write("explanation")

    col1, col2 = st.columns(2)

    # Setting column
    with col1:
        with st.container(border=True):
            st.subheader("Settings")

            mode = st.selectbox("Representation 2D/3D", [2, 3])
            model = st.selectbox("Prediction model", ["Rule-based", "Logistic Regression", "MLP"])
            draw = st.checkbox("Draw results overlay")
            source = st.segmented_control("Source", ["Upload file", "Demo file", "Record file"])

    # Source column
    with col2:
        with st.container(border=True):
            st.subheader("Source")

            match source:
                case "Upload file":
                    video_path = render_upload_source()

                case "Demo file":
                    video_path = render_demo_source()

                case "Record file":
                    video_path = render_record_source()

                case _:
                    st.info("Select a source in the settings tab.")
                    video_path = None

    if st.button("Run pipeline"):
        if video_path is None:
            st.warning("Please select or provide a source file first.")
            return

        with st.spinner("Processing video..."):
            output_video, results = run_video_pipeline(video_path, draw, model, mode)

        st.session_state.analyze_output_video = output_video
        st.session_state.analyze_results = results
        st.session_state.analyze_view = "results"
        st.toast("Video processed", icon="✅")
        st.rerun()

def render_results_view():
    output_video = st.session_state.analyze_output_video
    results = st.session_state.analyze_results

    col1, col2 = st.columns([1, 5])

    with col1:
        if st.button("← Back"):
            st.session_state.analyze_view = "input"
            st.rerun()

    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>✅ Result</h3>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="large")

        with col1:
            with st.container(border=True):
                st.metric("Repetitions", 10)
        with col2:
            with st.container(border=True):
                st.metric("Score", 90)
        with col3:
            with st.container(border=True):
                st.metric("Duration", "2 minutes")
        with col4:
            with st.container(border=True):
                st.metric("Avg rep time", "3,5 sec")

        if output_video is not None:
            with st.expander("Expand for video"):
                st.video(output_video)

                st.download_button(
                    "Download video",
                    data=output_video,
                    file_name="output.mp4",
                    mime="video/mp4"
                )

        if results is not None:
            with st.expander("Expand for detailed data"):
                st.dataframe(results)
                
# Source render helpers
@st.cache_resource
def get_pose_pipe():
    return MediaPipePose()

def render_record_source():
    record_path = os.path.join(tempfile.gettempdir(), "recorded_pushup.mp4")
    pipe = get_pose_pipe()

    webrtc_streamer(
        key="pushup-recorder",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: PoseProcessor(
            pipe=pipe,
            width=480,
            height=360,
            detect_every_n_frames=2,
        ),
        in_recorder_factory=lambda: MediaRecorder(record_path, format="mp4"),
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

    st.info("Start the camera, record your push-ups, then stop the stream.")

    if not os.path.exists(record_path):
        return None

    st.success("Recording saved")

    if st.button("Clear recording"):
        try:
            os.remove(record_path)
        except OSError:
            pass
        st.rerun()

    return record_path

def render_upload_source():
    return st.file_uploader("Upload file", type=["mp4"], max_upload_size = 50)

def render_demo_source() -> str | None:
    demo_files = {
        file.name: str(file)
        for file in sorted(Path("data/videos").glob("*.mp4"))
    }

    if not demo_files:
        st.warning("No demo videos found.")
        return None

    selected_demo = st.selectbox("Select demo file", list(demo_files.keys()))
    return demo_files[selected_demo]