from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class ModelIteration:
    tag: str
    label: str
    summary: str
    mae: str
    mae_delta: str | None
    r2: str
    r2_delta: str | None
    fnr: str
    fnr_delta: str | None
    recommended_t: int
    dropout: float
    tta_flip: bool
    checkpoint_candidates: tuple[str, ...]

    def resolve_checkpoint(self) -> Path | None:
        for filename in self.checkpoint_candidates:
            candidate = MODELS_DIR / filename
            if candidate.is_file():
                return candidate
        return None


ITERATIONS: tuple[ModelIteration, ...] = (
    ModelIteration(
        tag="r2",
        label="r2 · Partial fine-tune",
        summary="Titik awal. Hanya 3 blok akhir backbone yang dilatih dengan batch 1024, jadi ambang aman masih T = 31 dan hampir separuh orang dewasa tertahan.",
        mae="6.72 th",
        mae_delta=None,
        r2="0.769",
        r2_delta=None,
        fnr="46.0%",
        fnr_delta=None,
        recommended_t=31,
        dropout=0.2,
        tta_flip=False,
        checkpoint_candidates=("m1_r2.pt", "M1_s43_r2.pt"),
    ),
    ModelIteration(
        tag="r3",
        label="r3 · Full fine-tune",
        summary="Seluruh backbone dibuka (13 blok, BatchNorm ikut dilatih, batch 128). Kenaikan terbesar di anak dan remaja, T turun ke 24, tetapi mulai overfitting.",
        mae="5.66 th",
        mae_delta="-1.06 th",
        r2="0.830",
        r2_delta="+0.061",
        fnr="11.4%",
        fnr_delta="-34.6 poin",
        recommended_t=24,
        dropout=0.2,
        tta_flip=False,
        checkpoint_candidates=("m1_r3.pt", "M1_s43_r3.pt", "age_verification_model.pt"),
    ),
    ModelIteration(
        tag="r4",
        label="r4 · Regularisasi + TTA",
        summary="Weight decay 0.05, dropout 0.3, augmentasi rotasi, geser, dan skala, plus TTA flip. Overfitting berkurang dan usia 40+ paling membaik, tetapi latensi CPU naik ke 19 ms.",
        mae="5.11 th",
        mae_delta="-0.55 th",
        r2="0.859",
        r2_delta="+0.029",
        fnr="14.7%",
        fnr_delta="+3.3 poin",
        recommended_t=25,
        dropout=0.3,
        tta_flip=True,
        checkpoint_candidates=("m1_r4.pt", "M1_s43_r4.pt"),
    ),
)
