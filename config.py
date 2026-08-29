import os

# Set ENV=tc1 in SLURM scripts when running on cluster
# Defaults to local for Mac Air testing
ENV = os.environ.get('ENV', 'local')

if ENV == 'tc1':
    # Faye to fill in the correct TC1 paths
    DATA_REAL      = '/home/UG/friendusername/data/cifake/train/REAL'
    DATA_FAKE      = '/home/UG/friendusername/data/cifake/train/FAKE'
    DATA_TEST_REAL = '/home/UG/friendusername/data/cifake/test/REAL'
    DATA_TEST_FAKE = '/home/UG/friendusername/data/cifake/test/FAKE'
    VAL_REAL       = '/home/UG/friendusername/data/validation/real'
    VAL_AIGC       = '/home/UG/friendusername/data/validation/aigc'
    FEATURES_DIR   = '/home/UG/friendusername/features'
    MODEL_PATH     = '/home/UG/friendusername/classifier.pt'
else:
    # For local testing 
    DATA_REAL      = 'data_test/REAL'
    DATA_FAKE      = 'data_test/FAKE'
    DATA_TEST_REAL = 'data_test/REAL'
    DATA_TEST_FAKE = 'data_test/FAKE'
    VAL_REAL       = 'data_test/validation/real'
    VAL_AIGC       = 'data_test/validation/aigc'
    FEATURES_DIR   = 'features_test'
    MODEL_PATH     = 'classifier.pt'

# Model settings — everyone imports these
CLIP_DIM  = 512
DCT_DIM   = 64
INPUT_DIM = CLIP_DIM + DCT_DIM  # 576
BATCH_SIZE = 512
EPOCHS     = 50