import threading
import time
import streamlit as st
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer

from iterations import ITERATIONS
from live import LiveAgeProcessor
from model import crop_face, load_face_detector, load_model, predict_age

AGE_LIMIT = 18   # batas usia dewasa

@st.cache_resource
def get_model(path_str: str, dropout: float):
    return load_model(path_str, dropout=dropout)

@st.cache_resource
def get_face_detector():
    return load_face_detector(), threading.Lock()

# Project: Age Gating
st.title("Try Age Gating") # mw ganti kah kalo urusan urutan serah kalian deh ya pokoknya bagian ini jgn pake AI aja gw udah MUAKKKKK
st.write("This application simulates an age verification process using a camera input. Please take a picture of yourself to verify your age.") # gak tau yapping

iterations_by_tag = {it.tag: it for it in ITERATIONS}
available_tags = [it.tag for it in ITERATIONS if it.resolve_checkpoint() is not None]

selected_tag = st.segmented_control(
    "Model iteration",
    list(iterations_by_tag),
    format_func=lambda tag: iterations_by_tag[tag].label,
    default=available_tags[-1] if available_tags else ITERATIONS[-1].tag,
    required=True,
)
iteration = iterations_by_tag[selected_tag]
checkpoint = iteration.resolve_checkpoint()

# Simulate a delay for the camera input
st.subheader("Try Age Verification")

if checkpoint is None:
    st.info(f"Checkpoint {iteration.tag} belum ada di `age-gating/models/`. Pilih iterasi lain untuk mencoba.", icon=":material/info:")
else:
    model = get_model(str(checkpoint), iteration.dropout)

    input_method = st.segmented_control(
        "Photo source",
        [":material/photo_camera: Camera", ":material/videocam: Live", ":material/upload: Upload"],
        default=":material/photo_camera: Camera",
        required=True,
    )

    picture = None

    if input_method == ":material/photo_camera: Camera":
        placeholder = st.empty()

        if "camera_ready" not in st.session_state:
            placeholder.skeleton(height=400)
            time.sleep(2)  # Simulate camera loading time
            st.session_state.camera_ready = True
            placeholder.empty()

        picture = st.camera_input("Take a picture of yourself to verify your age.")
    elif input_method == ":material/videocam: Live":
        st.caption("Press start and look at the camera. The estimate is the median of the last 2 seconds, and the decision appears once it is stable.")
        webrtc_streamer(
            key=f"live-age-{iteration.tag}",
            video_processor_factory=lambda: LiveAgeProcessor(model, AGE_LIMIT, tta_flip=iteration.tta_flip),
            media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        )
    else:
        picture = st.file_uploader("Upload a photo of your face", type=["jpg", "jpeg", "png"])
        if picture:
            st.image(picture, width=300, alt="Uploaded face photo")

    if picture:
        face_detector, detector_lock = get_face_detector()
        with st.status("Verifying age based on your picture...") as status:
            img = ImageOps.exif_transpose(Image.open(picture))
            with detector_lock:
                face = crop_face(face_detector, img)

            if face is None:
                status.update(label="No face detected", state="error", expanded=False)
            else:
                age = round(predict_age(model, face, tta_flip=iteration.tta_flip))
                st.image(face, caption="Face read by the model", width=160, alt="Cropped face used for age estimation")
                status.update(label="Age verification complete!", state="complete", expanded=False)

        if face is None:
            st.warning("No face detected. Make sure your face is clearly visible, facing the camera, and well lit, then try again.")
        else:
            is_adult = age >= AGE_LIMIT

            if is_adult:
                st.success(f"Your age has been successfully verified ({age} years old). You are allowed to access age-restricted content.")
            else:
                st.error(f"Your age has been verified ({age} years old). You are not allowed to access age-restricted content.")

st.divider()

st.subheader("What changed")
st.write(iteration.summary)

with st.container(horizontal=True):
    st.metric("MAE validasi", iteration.mae, iteration.mae_delta, delta_color="inverse", border=True)
    st.metric("R²", iteration.r2, iteration.r2_delta, border=True)
    st.metric(f"Dewasa tertahan (T = {iteration.recommended_t})", iteration.fnr, iteration.fnr_delta, delta_color="inverse", border=True)
