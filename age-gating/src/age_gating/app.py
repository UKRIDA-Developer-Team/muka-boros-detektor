import time
import streamlit as st
from PIL import Image, ImageOps

from streamlit_webrtc import webrtc_streamer

from age_gating.live import LiveAgeProcessor
from age_gating.model import crop_face, load_face_detector, load_model, predict_age

AGE_LIMIT = 18   # batas usia dewasa

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_face_detector():
    return load_face_detector()

model = get_model()
face_detector = get_face_detector()

# Project: Age Gating
st.title("Try Age Gating") # mw ganti kah kalo urusan urutan serah kalian deh ya pokoknya bagian ini jgn pake AI aja gw udah MUAKKKKK
st.write("This application simulates an age verification process using a camera input. Please take a picture of yourself to verify your age.") # gak tau yapping

# Simulate a delay for the camera input
st.subheader("Try Age Verification")

input_method = st.segmented_control(
    "Photo source",
    [":material/photo_camera: Camera", ":material/upload: Upload", ":material/videocam: Live"],
    default=":material/photo_camera: Camera",
    required=True,
)

if input_method == ":material/photo_camera: Camera":
    placeholder = st.empty()

    if "camera_ready" not in st.session_state:
        placeholder.skeleton(height=400)
        time.sleep(2)  # Simulate camera loading time
        st.session_state.camera_ready = True
        placeholder.empty()

    picture = st.camera_input("Take a picture of yourself to verify your age.")
elif input_method == ":material/upload: Upload":
    picture = st.file_uploader("Upload a photo of your face", type=["jpg", "jpeg", "png"])
    if picture:
        st.image(picture, width=300, alt="Uploaded face photo")
else:
    picture = None
    st.caption("Press start and look at the camera. The estimate is the median of the last 2 seconds, and the decision appears once it is stable.")
    webrtc_streamer(
        key="live-age",
        video_processor_factory=lambda: LiveAgeProcessor(model, AGE_LIMIT),
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
    )

if picture:
    with st.status("Verifying age based on your picture...") as status:
        img = ImageOps.exif_transpose(Image.open(picture))
        face = crop_face(face_detector, img)

        if face is None:
            status.update(label="No face detected", state="error", expanded=False)
            st.warning("Make sure your face is clearly visible, facing the camera, and well lit, then try again.", title="No face detected")
            st.stop()

        age = round(predict_age(model, face))
        st.image(face, caption="Face read by the model", width=160, alt="Cropped face used for age estimation")

        status.update(label="Age verification complete!", state="complete", expanded=False)

    is_adult = age >= AGE_LIMIT

    if is_adult:
        st.success(f"Your age has been successfully verified ({age} years old). You are allowed to access age-restricted content.")
    else:
        st.error(f"Your age has been verified ({age} years old). You are not allowed to access age-restricted content.")
