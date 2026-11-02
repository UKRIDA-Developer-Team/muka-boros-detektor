import time
import streamlit as st
from PIL import Image

from age_gating.model import load_model, predict_age

AGE_LIMIT = 18   # batas usia dewasa

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Project: Age Gating
st.title("Try Age Gating") # mw ganti kah kalo urusan urutan serah kalian deh ya pokoknya bagian ini jgn pake AI aja gw udah MUAKKKKK
st.write("This application simulates an age verification process using a camera input. Please take a picture of yourself to verify your age.") # gak tau yapping

# Simulate a delay for the camera input
st.subheader("Try Age Verification")

input_method = st.segmented_control(
    "Photo source",
    [":material/photo_camera: Camera", ":material/upload: Upload"],
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
else:
    picture = st.file_uploader("Upload a photo of your face", type=["jpg", "jpeg", "png"])
    if picture:
        st.image(picture, width=300, alt="Uploaded face photo")

if picture:
    with st.status("Verifying age based on your picture...") as status:
        img = Image.open(picture)
        age = round(predict_age(model, img))

        status.update(label="Age verification complete!", state="complete", expanded=False)

    is_adult = age >= AGE_LIMIT

    if is_adult:
        st.success(f"Your age has been successfully verified ({age} years old). You are allowed to access age-restricted content.")
    else:
        st.error(f"Your age has been verified ({age} years old). You are not allowed to access age-restricted content.")
