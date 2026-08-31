import os
from pathlib import Path

# ================================================================
# ONE CONFIG FILE FOR BOTH LOCAL + GPU
# Local laptop:
#   ENV is not set -> defaults to "local"
# TC1 / GPU:
#   jobs/job_extract.sh, jobs/job_train.sh, jobs/job_predict.sh
#   should contain:
#       export ENV=tc1
# ================================================================

ENV = os.environ.get("ENV", "local").lower()

PROJECT_ROOT = Path(__file__).resolve().parent

# LOCAL LAPTOP TEST
if ENV == "local":
    TRAIN_SOURCES = [
        {
            "name": "sid",
            "type": "hf",
            "repo": "saberzl/SID_Set",
            "split": "train",
            "max_per_class": 100,   # LOCAL ONLY: tiny cap for smoke test
            "n_aug": 0,
            "shuffle_buffer": 100,
        },
        {
            "name": "cifake_real",
            "type": "dir",
            "path": str(PROJECT_ROOT / "data_test" / "REAL"),
            "label": 0,
            "shuffle": False,
            "max_per_class": 100,   # LOCAL ONLY: tiny cap for smoke test
            "n_aug": 0,
        },
        {
            "name": "cifake_fake",
            "type": "dir",
            "path": str(PROJECT_ROOT / "data_test" / "FAKE"),
            "label": 1,
            "shuffle": False,
            "max_per_class": 100,   # LOCAL ONLY: tiny cap for smoke test
            "n_aug": 0,
        },
        {
            "name": "synthbuster",
            "type": "dir",
            "path": str(PROJECT_ROOT / "data_test" / "synthbuster"),
            "max_per_class": 20,      # LOCAL ONLY: tiny cap for smoke test
            "n_aug": 0,
        },
        {
            "name": "tiny_genimage",
            "type": "hf",
            "repo": "TheKernel01/Tiny-GenImage",
            "split": "train",
            "image_key": "image",
            "label_key": "label",
            "max_per_class": 20,      # LOCAL ONLY: tiny cap for smoke test
            "n_aug": 0,
            "shuffle_buffer": 100,
        },
    ]

    VAL_SOURCE_NAMES = []

    FEATURES_TRAIN = str(PROJECT_ROOT / "features_test")
    MODEL_PATH = str(PROJECT_ROOT / "results" / "classifier.pt")

    PREDICT_IMAGES = str(PROJECT_ROOT / "input")
    PREDICTIONS_OUT = str(PROJECT_ROOT / "results" / "predictions_local.json")
    ROBUSTNESS_OUT = str(PROJECT_ROOT / "results" / "robustness_local.json")

    MAX_PER_CLASS = None
    N_AUG = 0
    N_WORKERS = 1
    EXTRACT_BATCH_SIZE = 8
    SHARD_SIZE = 1000

    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-3

# TC1 / GPU FULL RUN
elif ENV == "tc1":

    GPU_DATA_ROOT = Path("/tc1home/FYP/faye0004/techjam-track5/data")
    GPU_OUTPUT_ROOT = Path("/tc1home/FYP/faye0004/track5")

    TRAIN_SOURCES = [
        {
            "name": "sid",
            "type": "hf",
            "repo": "saberzl/SID_Set",
            "split": "train",
            "max_per_class": None,
            "n_aug": 1,
            "shuffle_buffer": 10000,
        },
        {
            "name": "cifake",
            "type": "dir",
            "path": str(GPU_DATA_ROOT / "cifake" / "train"),
            "max_per_class": None,
            "n_aug": 1,
        },
        {
            "name": "cifake_test",
            "type": "dir",
            "path": str(GPU_DATA_ROOT / "cifake" / "test"),
            "max_per_class": None,
            "n_aug": 0,
        },
        {
            # SynthBuster: 9k images across 9 generators including
            "name": "synthbuster",
            "type": "dir",
            "path": str(GPU_DATA_ROOT / "synthbuster32" / "resized_data_Synthbuster" / "Synthbuster_Dataset"),
            "max_per_class": None,
            "n_aug": 1,
        },
        {
            # Tiny-GenImage: curated subset of GenImage.
            "name": "tiny_genimage",
            "type": "hf",
            "repo": "TheKernel01/Tiny-GenImage",
            "split": "train",
            "image_key": "image",
            "label_key": "label",
            "max_per_class": None,
            "n_aug": 1,
            "shuffle_buffer": 10000,
        },
    ]

    VAL_SOURCE_NAMES = ["cifake_test"]

    FEATURES_TRAIN = str(GPU_OUTPUT_ROOT / "feature_test")
    MODEL_PATH = str(GPU_OUTPUT_ROOT / "results" / "classifier.pt")

    PREDICT_IMAGES = str(GPU_DATA_ROOT / "validation")
    PREDICTIONS_OUT = str(GPU_OUTPUT_ROOT / "results" / "predictions.json")
    ROBUSTNESS_OUT = str(GPU_OUTPUT_ROOT / "results" / "robustness_summary.json")

    MAX_PER_CLASS = None
    N_AUG = 1
    N_WORKERS = 4
    EXTRACT_BATCH_SIZE = 64
    SHARD_SIZE = 20000

    BATCH_SIZE = 512
    EPOCHS = 75    # bumped from 50 — L/14 features + larger dataset needs more epochs
    LR = 1e-3


else:
    raise ValueError(
        f"Unknown ENV={ENV!r}. Expected 'local' or 'tc1'."
    )


# SHARED
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"

SEED = 42
VAL_FRACTION = 0.15

CLIP_DIM = 768  
DCT_DIM = 64
INPUT_DIM = CLIP_DIM + DCT_DIM  # 832 total 