import time
import streamlit as st

age = 16
is_adult = age >= 18

# Project: Age Gating
st.title("Try Age Gating") # mw ganti kah kalo urusan urutan serah kalian deh ya pokoknya bagian ini jgn pake AI aja gw udah MUAKKKKK
st.write("This application simulates an age verification process using a camera input. Please take a picture of yourself to verify your age.") # gak tau yapping

# Simulate a delay for the camera input
st.subheader("Try Age Verification")
placeholder = st.empty()

if "camera_ready" not in st.session_state:
    placeholder.skeleton(height=400)
    time.sleep(2)  # Simulate camera loading time
    st.session_state.camera_ready = True
    placeholder.empty()

picture = st.camera_input("Take a picture of yourself to verify your age.")

if picture:
    # hardcoded age verification logic for demonstration purposes
    age = 16  # Replace this with actual age verification logic based on the picture
    is_adult = age >= 18

    with st.status("Verifying age based on your picture...") as status:
        time.sleep(2)  # Simulate age verification process
        status.update(label="Age verification complete!", state="complete", expanded=False)

    if is_adult:
        st.success("Your age has been successfully verified. You are allowed to access age-restricted content.")
    else:
        st.error("Your age has been verified. You are not allowed to access age-restricted content.")
