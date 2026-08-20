import os
from pathlib import Path

## Test Options
TEST_METHOD = 'DARE' # It's only DARE
DATA_TYPE = 'node'  # 'map', 'node'
USE_TEST_DATASET = True  # False for train dataset, True for test dataset
USE_DELTA_POSITION = True # False for absolute position, True for delta position
USE_EXPLORATION_RATE_FOR_DONE = False  # False for robot util == 0, True for exploration rate
TEST_N_AGENTS = 1 # SINGLE AGENT keep it 1

## Environment Runner Options
USE_GPU = True  # do you want to use GPUS?
NUM_GPU = 1  # the number of GPUs
NUM_META_AGENT = 10  # the number of processes

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = True  # do you want to save GIFs

ACTION_HORIZON = None # None for 1 horizon

## Name and Path
# 中文目的：原始 DARE 测试和 MergingMap 论文消融可通过同一环境变量使用同一检查点。
# English purpose: let original-DARE and MergingMap tests share one checkpoint override.
_DEFAULT_RUN_PATH = Path(
    '/root/lite dare/DARE/lite_dare/runs/'
    'DiscreteLiteDARE_NodeEncSA_L4_H4_D256_DecL1_H4/'
    'seed_42_20260726_004608'
)
_DEFAULT_CHECKPOINT_NAME = 'epoch=0150-val_loss=0.058.ckpt'
_checkpoint_override = os.environ.get('DARE_CHECKPOINT_PATH')
if _checkpoint_override:
    _checkpoint_path = Path(_checkpoint_override).expanduser().resolve()
    run_path = str(_checkpoint_path.parent.parent)
    checkpoint_name = _checkpoint_path.name
else:
    run_path = str(_DEFAULT_RUN_PATH)
    checkpoint_name = _DEFAULT_CHECKPOINT_NAME
gifs_path = f'{run_path}/gifs'