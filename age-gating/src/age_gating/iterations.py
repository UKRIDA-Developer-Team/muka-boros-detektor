from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
SAMPLE_IMAGE_PATH = ROOT_DIR / "assets" / "sample_face.png"


@dataclass(frozen=True)
class ModelIteration:
    tag: str
    tab_label: str
    summary: str
    config_note: str
    val_mae: str
    val_mae_delta: str
    val_mae_delta_color: str
    mse: str
    mse_delta: str
    mse_delta_color: str
    r2: str
    r2_delta: str
    r2_delta_color: str
    fnr: str
    fnr_delta: str
    fnr_delta_color: str
    recommended_t: int
    dropout: float
    tta_flip: bool
    checkpoint_candidates: tuple[str, ...]
    expected_checkpoint: str
    reference_age: float | None
    example_note: str

    def resolve_checkpoint(self) -> Path | None:
        for filename in self.checkpoint_candidates:
            candidate = MODELS_DIR / filename
            if candidate.is_file():
                return candidate
        return None


ITERATIONS: tuple[ModelIteration, ...] = (
    ModelIteration(
        tag="r2",
        tab_label="r2 · Partial fine-tune",
        summary="Fine-tuning hanya 3 blok akhir backbone dengan batch besar (1024). Ambang aman T masih tinggi sehingga 46.0% dewasa tertahan.",
        config_note="UNFREEZE_LAST=3 · BATCH_SIZE=1024 · LR_BACKBONE=1e-5 · BN beku",
        val_mae="6.72 th",
        val_mae_delta="Baseline",
        val_mae_delta_color="off",
        mse="88.00",
        mse_delta="Baseline",
        mse_delta_color="off",
        r2="0.769",
        r2_delta="Baseline",
        r2_delta_color="off",
        fnr="46.0%",
        fnr_delta="T = 31 th",
        fnr_delta_color="off",
        recommended_t=31,
        dropout=0.2,
        tta_flip=False,
        checkpoint_candidates=("m1_r2.pt", "M1_s43_r2.pt"),
        expected_checkpoint="models/m1_r2.pt",
        reference_age=None,
        example_note="Di r2, ambang aman berada di T = 31 th (FPR ≤ 5%), sehingga pengguna dewasa muda kerap tertahan verifikasi tambahan.",
    ),
    ModelIteration(
        tag="r3",
        tab_label="r3 · Full fine-tune",
        summary="Seluruh 13 blok backbone dibuka dengan batch 128 dan pembaruan BatchNorm. MAE turun 15.8% dan FNR dewasa anjlok dari 46.0% ke 11.4%.",
        config_note="UNFREEZE_LAST=13 · BATCH_SIZE=128 · LR_BACKBONE=1e-4 · BN aktif · Warmup 2 ep",
        val_mae="5.66 th",
        val_mae_delta="-1.06 th (-15.8%)",
        val_mae_delta_color="inverse",
        mse="64.69",
        mse_delta="-23.31 (-26.5%)",
        mse_delta_color="inverse",
        r2="0.830",
        r2_delta="+0.061 vs r2",
        r2_delta_color="normal",
        fnr="11.4%",
        fnr_delta="-34.6 poin · T = 24 th",
        fnr_delta_color="inverse",
        recommended_t=24,
        dropout=0.2,
        tta_flip=False,
        checkpoint_candidates=("m1_r3.pt", "M1_s43_r3.pt", "age_verification_model.pt"),
        expected_checkpoint="models/m1_r3.pt",
        reference_age=17.39,
        example_note="Membuka seluruh backbone menurunkan galat usia muda secara drastis dan menekan ambang T dari 31 ke 24 th, walau overfitting mulai naik.",
    ),
    ModelIteration(
        tag="r4",
        tab_label="r4 · Reg + TTA",
        summary="Regularisasi lebih kuat (weight decay 0.05, dropout 0.3), augmentasi rotasi/skala, dan TTA flip. Overfitting turun dan MAE membaik 9.8%.",
        config_note="WEIGHT_DECAY=0.05 · DROPOUT=0.3 · Augmentasi rotasi/skala · TTA_FLIP=True",
        val_mae="5.11 th",
        val_mae_delta="-0.55 th (-9.8%)",
        val_mae_delta_color="inverse",
        mse="53.63",
        mse_delta="-11.06 (-17.1%)",
        mse_delta_color="inverse",
        r2="0.859",
        r2_delta="+0.029 vs r3",
        r2_delta_color="normal",
        fnr="14.7%",
        fnr_delta="+3.3 poin · T = 25 th",
        fnr_delta_color="inverse",
        recommended_t=25,
        dropout=0.3,
        tta_flip=True,
        checkpoint_candidates=("m1_r4.pt", "M1_s43_r4.pt"),
        expected_checkpoint="models/m1_r4.pt",
        reference_age=22.27,
        example_note="Pada foto uji notebook, prediksi bergeser dari ~17 th (r3) menjadi ~22 th (r4) dengan estimasi yang lebih stabil berkat TTA flip.",
    ),
)
