import os
from pathlib import Path

# ================================================================
# ONE CONFIG FILE FOR BOTH LOCAL + GPU
#
# You do NOT manually switch configs.
#
# Local laptop:
#   ENV is not set -> defaults to "local"
#
# TC1 / GPU:
#   jobs/job_extract.sh, jobs/job_train.sh, jobs/job_predict.sh
#   should contain:
#       export ENV=tc1
#
# So the same config.py automatically selects the correct setup.
# ================================================================

ENV = os.environ.get("ENV", "local").lower()

PROJECT_ROOT = Path(__file__).resolve().parent


# ================================================================
# LOCAL LAPTOP TEST
# ================================================================
if ENV == "local":

    # Local CIFAKE subset already present in this repo:
    #
    # data_test/
    #   REAL/
    #   FAKE/
    #
    # We use every image that exists in those two local subset folders.
    #
    # SID is streamed from Hugging Face, but capped to only 100 examples
    # from each RAW SID class for the local smoke test:
    #   raw 0 = real
    #   raw 1 = fully synthetic
    #   raw 2 = tampered
    #
    # stream_extract.py maps these to the binary Track-5 target:
    #   0 -> authentic
    #   1 -> AIGC
    #   2 -> AIGC

    TRAIN_SOURCES = [
        {
            "name": "sid",
            "type": "hf",
            "repo": "saberzl/SID_Set",
            "split": "train",
            "max_per_class": 100,     # LOCAL ONLY: 100 per raw SID class
            "n_aug": 0,
            "shuffle_buffer": 100,    # small buffer so local test starts fast
        },
        {
            "name": "cifake_real",
            "type": "dir",
            "path": str(PROJECT_ROOT / "data_test" / "REAL"),
            "label": 0,
            "shuffle": False,
            "max_per_class": 100,     # LOCAL ONLY: top 100 REAL images
            "n_aug": 0,
        },
        {
            "name": "cifake_fake",
            "type": "dir",
            "path": str(PROJECT_ROOT / "data_test" / "FAKE"),
            "label": 1,
            "shuffle": False,
            "max_per_class": 100,     # LOCAL ONLY: top 100 FAKE images
            "n_aug": 0,
        },
    ]

    # No external validation dataset during the smoke test.
    # train.py will use its grouped random split.
    VAL_SOURCE_NAMES = []

    FEATURES_TRAIN = str(PROJECT_ROOT / "features_test")
    MODEL_PATH = str(PROJECT_ROOT / "results" / "classifier_local.pt")

    # Defined for compatibility; WildFake is not part of local training.
    PREDICT_IMAGES = str(PROJECT_ROOT / "data_test" / "validation")
    PREDICTIONS_OUT = str(PROJECT_ROOT / "results" / "predictions_local.json")
    ROBUSTNESS_OUT = str(PROJECT_ROOT / "results" / "robustness_local.json")

    # Small/local settings
    MAX_PER_CLASS = None
    N_AUG = 0
    N_WORKERS = 1
    EXTRACT_BATCH_SIZE = 8
    SHARD_SIZE = 1000

    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-3


# ================================================================
# TC1 / GPU FULL RUN
# ================================================================
elif ENV == "tc1":

    # Full CIFAKE dataset already stored on TC1.
    GPU_DATA_ROOT = Path("/tc1home/FYP/faye0004/techjam-track5/data")

    # Feature/model outputs.
    GPU_OUTPUT_ROOT = Path("/tc1home/FYP/faye0004/track5")

    TRAIN_SOURCES = [
        {
            "name": "sid",
            "type": "hf",
            "repo": "saberzl/SID_Set",
            "split": "train",

            # GPU FULL RUN:
            # None means consume the entire SID train stream.
            # No 20-image cap and no 30k cap.
            "max_per_class": None,

            # Clean + one randomly selected Track-5 transform per image.
            "n_aug": 1,

            # Larger shuffle buffer is fine on the cluster.
            "shuffle_buffer": 10000,
        },
        {
            "name": "cifake",
            "type": "dir",
            "path": str(GPU_DATA_ROOT / "cifake" / "train"),

            # Use ALL CIFAKE train images present on TC1.
            "max_per_class": None,
            "n_aug": 1,
        },
        {
            "name": "cifake_test",
            "type": "dir",
            "path": str(GPU_DATA_ROOT / "cifake" / "test"),

            # Use ALL CIFAKE test images as internal validation.
            "max_per_class": None,
            "n_aug": 0,
        },
    ]

    # CIFAKE test is kept out of optimisation and used for model selection.
    # WildFake is NOT included here.
    VAL_SOURCE_NAMES = ["cifake_test"]

    FEATURES_TRAIN = str(GPU_OUTPUT_ROOT / "feature_test")
    MODEL_PATH = str(GPU_OUTPUT_ROOT / "results" / "classifier.pt")

    # WildFake is used only AFTER the model has been trained/frozen.
    PREDICT_IMAGES = str(GPU_DATA_ROOT / "validation")
    PREDICTIONS_OUT = str(GPU_OUTPUT_ROOT / "results" / "predictions.json")
    ROBUSTNESS_OUT = str(GPU_OUTPUT_ROOT / "results" / "robustness_summary.json")

    # Full GPU extraction settings
    MAX_PER_CLASS = None
    N_AUG = 1
    N_WORKERS = 4
    EXTRACT_BATCH_SIZE = 64
    SHARD_SIZE = 20000

    # Full classifier training settings
    BATCH_SIZE = 512
    EPOCHS = 50
    LR = 1e-3


else:
    raise ValueError(
        f"Unknown ENV={ENV!r}. Expected 'local' or 'tc1'."
    )


# ================================================================
# SHARED
# ================================================================
# Shared OpenCLIP configuration used by extraction and training metadata.
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

SEED = 42
VAL_FRACTION = 0.15

CLIP_DIM = 512
DCT_DIM = 64
INPUT_DIM = CLIP_DIM + DCT_DIM
