import base64
import io
import time
import streamlit as st
from PIL import Image

from age_gating.iterations import ITERATIONS, SAMPLE_IMAGE_PATH
from age_gating.live_camera import live_camera
from age_gating.model import load_model, predict_age

AGE_LIMIT = 18   # batas usia dewasa
LIVE_COMPONENT_KEY = "age_gating_live_widget"


@st.cache_resource
def get_model(path_str: str, dropout: float):
    return load_model(path_str, dropout=dropout)


@st.cache_data
def get_sample_image() -> Image.Image:
    with Image.open(SAMPLE_IMAGE_PATH) as image:
        return image.convert("RGB").copy()


def run_prediction(iteration, checkpoint_path, image: Image.Image) -> tuple[float, float]:
    model = get_model(str(checkpoint_path), iteration.dropout)
    start = time.perf_counter()
    raw_age = predict_age(model, image, tta_flip=iteration.tta_flip)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return raw_age, elapsed_ms


def decode_data_url(data_url: str) -> Image.Image:
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        return image.convert("RGB").copy()


@st.fragment
def render_live_section(iteration, checkpoint_path):
    def handle_frame():
        component_state = st.session_state.get(LIVE_COMPONENT_KEY)
        payload = component_state.get("frame") if component_state else None
        if not payload or not payload.get("image"):
            return
        img = decode_data_url(payload["image"])
        raw_age, elapsed_ms = run_prediction(iteration, checkpoint_path, img)
        age = round(raw_age)
        st.session_state["live_result"] = {
            "seq": payload.get("seq", 0),
            "modelTag": iteration.tag,
            "age": age,
            "rawAge": round(raw_age, 2),
            "isAdult": age >= AGE_LIMIT,
            "meetsT": raw_age >= iteration.recommended_t,
            "elapsedMs": round(elapsed_ms, 1),
        }

    def handle_stopped():
        st.session_state.pop("live_result", None)

    current_result = st.session_state.get("live_result")
    if current_result and current_result.get("modelTag") != iteration.tag:
        current_result = None

    live_camera(
        model_tag=iteration.tag,
        recommended_t=iteration.recommended_t,
        age_limit=AGE_LIMIT,
        result=current_result,
        key=LIVE_COMPONENT_KEY,
        on_frame_change=handle_frame,
        on_stopped_change=handle_stopped,
    )

    if current_result:
        age = current_result["age"]
        if current_result["isAdult"]:
            st.success(f"Your age has been successfully verified ({age} years old). You are allowed to access age-restricted content.")
        else:
            st.error(f"Your age has been verified ({age} years old). You are not allowed to access age-restricted content.")

        gating_status = (
            "Lolos otomatis"
            if current_result["meetsT"]
            else "Perlu verifikasi tambahan"
        )
        st.caption(
            f"Live {iteration.tag} · prediksi mentah {current_result['rawAge']:.2f} th · CPU {current_result['elapsedMs']:.1f} ms · ambang T = {iteration.recommended_t} th ({gating_status})"
        )


# Project: Age Gating
st.title("Try Age Gating") # mw ganti kah kalo urusan urutan serah kalian deh ya pokoknya bagian ini jgn pake AI aja gw udah MUAKKKKK
st.write("This application simulates an age verification process using a camera input. Please take a picture of yourself to verify your age.") # gak tau yapping

available_tabs = [it.tab_label for it in ITERATIONS if it.resolve_checkpoint() is not None]
default_tab = available_tabs[-1] if available_tabs else ITERATIONS[-1].tab_label

tabs = st.tabs(
    [it.tab_label for it in ITERATIONS],
    default=default_tab,
    key="model_iteration_tabs",
    on_change="rerun",
)

active_iteration = ITERATIONS[-1]
active_checkpoint = None

for tab, iteration in zip(tabs, ITERATIONS):
    if not tab.open:
        continue

    active_iteration = iteration
    active_checkpoint = iteration.resolve_checkpoint()

    with tab:
        if active_checkpoint is not None:
            st.badge(f"Live · {iteration.tag}", icon=":material/check_circle:", color="green")
        else:
            st.badge(f"Referensi notebook · {iteration.tag}", icon=":material/info:", color="gray")

st.subheader(f"Try Age Verification ({active_iteration.tag})")

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
    if active_checkpoint is None:
        st.info(
            f"Checkpoint untuk **{active_iteration.tag}** belum tersedia di `age-gating/{active_iteration.expected_checkpoint}`. "
            f"Pilih tab dengan status **Live** atau salin file `.pt` hasil pelatihan untuk menjalankan verifikasi live.",
            icon=":material/info:",
        )
    else:
        render_live_section(active_iteration, active_checkpoint)
else:
    picture = st.file_uploader("Upload a photo of your face", type=["jpg", "jpeg", "png"])
    if picture:
        st.image(picture, width=300, alt="Uploaded face photo")

if picture:
    if active_checkpoint is None:
        st.info(
            f"Checkpoint untuk **{active_iteration.tag}** belum tersedia di `age-gating/{active_iteration.expected_checkpoint}`. "
            f"Pilih tab dengan status **Live** atau salin file `.pt` hasil pelatihan untuk menjalankan verifikasi langsung.",
            icon=":material/info:",
        )
    else:
        with st.status(f"Verifying age with {active_iteration.tag}...") as status:
            img = Image.open(picture)
            raw_age, elapsed_ms = run_prediction(active_iteration, active_checkpoint, img)
            age = round(raw_age)

            status.update(label="Age verification complete!", state="complete", expanded=False)

        is_adult = age >= AGE_LIMIT

        if is_adult:
            st.success(f"Your age has been successfully verified ({age} years old). You are allowed to access age-restricted content.")
        else:
            st.error(f"Your age has been verified ({age} years old). You are not allowed to access age-restricted content.")

        st.caption(
            f"Model {active_iteration.tag} · prediksi mentah {raw_age:.2f} th · CPU {elapsed_ms:.1f} ms · ambang rekomendasi T = {active_iteration.recommended_t} th"
        )

st.divider()

st.subheader(f"Performa & perubahan ({active_iteration.tag})")
st.write(active_iteration.summary)
st.caption(active_iteration.config_note)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "MAE validasi",
    active_iteration.val_mae,
    active_iteration.val_mae_delta,
    delta_color=active_iteration.val_mae_delta_color,
    border=True,
    height="stretch",
)
c2.metric(
    "MSE",
    active_iteration.mse,
    active_iteration.mse_delta,
    delta_color=active_iteration.mse_delta_color,
    border=True,
    height="stretch",
)
c3.metric(
    "R²",
    active_iteration.r2,
    active_iteration.r2_delta,
    delta_color=active_iteration.r2_delta_color,
    border=True,
    height="stretch",
)
c4.metric(
    "Dewasa tertahan",
    active_iteration.fnr,
    active_iteration.fnr_delta,
    delta_color=active_iteration.fnr_delta_color,
    border=True,
    height="stretch",
)

with st.container(border=True):
    st.caption("Contoh langsung perubahan antar iterasi")
    img_col, res_col = st.columns([1, 2], vertical_alignment="center")
    sample_img = get_sample_image()
    img_col.image(
        sample_img,
        width=160,
        alt="Foto wajah contoh untuk melihat perubahan prediksi tiap iterasi model",
    )
    with res_col:
        if active_checkpoint is not None:
            raw_age, elapsed_ms = run_prediction(active_iteration, active_checkpoint, sample_img)
            shown_age = round(raw_age)
            gating_status = (
                "Lolos otomatis"
                if raw_age >= active_iteration.recommended_t
                else "Perlu verifikasi tambahan"
            )
            st.markdown(f"**Prediksi live ({active_iteration.tag}):** {shown_age} tahun (`{raw_age:.2f}` th)")
            st.caption(
                f"Batas {AGE_LIMIT} th · Ambang T = {active_iteration.recommended_t} th ({gating_status}) · CPU {elapsed_ms:.1f} ms"
            )
        elif active_iteration.reference_age is not None:
            shown_age = round(active_iteration.reference_age)
            gating_status = (
                "Lolos otomatis"
                if active_iteration.reference_age >= active_iteration.recommended_t
                else "Perlu verifikasi tambahan"
            )
            st.markdown(f"**Hasil notebook ({active_iteration.tag}):** {shown_age} tahun (`{active_iteration.reference_age:.2f}` th)")
            st.caption(
                f"Ambang T = {active_iteration.recommended_t} th ({gating_status}) · Taruh `{active_iteration.expected_checkpoint}` untuk inferensi live"
            )
        else:
            st.markdown(f"**Checkpoint `{active_iteration.tag}` belum dipasang**")
            st.caption(
                f"Ambang T = {active_iteration.recommended_t} th · Taruh `{active_iteration.expected_checkpoint}` untuk melihat prediksi live pada foto ini"
            )
        st.write(active_iteration.example_note)
