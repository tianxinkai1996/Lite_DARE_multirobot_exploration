# LiteDARE Based Multi-Robot Collaborative Exploration under Local and Low Bandwidth Communication

# This project is based on DARE
# GitHub



## 1. Dataset Preparation

Must edit `dataset_parameter.py` to set parameters for separate Train and test sets, then run:

```bash
python dataset_driver.py
```

- Datasets are stored as Zarr under `dataset/{method}_{type}_{train|test}_{count}`.
---

## 2. Model Training

### 2.1 Original DARE (L6 reference)


```bash

# Controlled comparison in the thesis uses batch=32; adjust via Hydra override
python train.py --config-name=train_exploration_transformer_node_discrete.yaml \
    dataloader.batch_size=32 val_dataloader.batch_size=16
```

Run directories are written to `runs/{date}/{time}_{name}_{task}/`. Checkpoints are saved as
`epoch={epoch:04d}-val_loss={val_loss:.3f}.ckpt` and selected by **minimum validation loss**


### 2.2 LiteDARE (L4 / L2, retrained from scratch)

```bash
# LiteDARE-L4 (4 graph self-attention layers)
python lite_dare/train_lite_dare.py --encoder-layers 4 --seed 42 --epochs 200 \
    --train-batch-size 32 --val-batch-size 16 --num-workers 0

# LiteDARE-L2 (2 graph self-attention layers)
python lite_dare/train_lite_dare.py --encoder-layers 2 --seed 42 --epochs 200 \
    --train-batch-size 32 --val-batch-size 16 --num-workers 0

```

The script automatically:
1. Swaps the graph encoder for `LiteExplorationNodeEncoder`, keeping the decoder / diffusion Transformer unchanged;
2. Forces `resume=False` (clean retraining from scratch);
3. Writes to `lite_dare/runs/{architecture}/seed_{seed}_{timestamp}/`.

Alternatively, `python lite_dare/run_lite_ablation.py` trains L4 and L2 in sequence (check its argument conventions first).

---

## 3. Model Testing

### 3.1 Single-robot testing

First edit `test_parameter.py`:
- `run_path`: set lite dare checkpoint paths
- `NUM_TEST` (default 100), `USE_GPU`, `NUM_META_AGENT`, `SAVE_GIFS`

```bash
python test_driver.py
```

### 3.2 Multi-robot testing

```bash
# All 100 test maps, teams 2/4/6/8, compressed communication, coordination enabled
python multi_test_driver.py --maps all --map-count 100 \
    --team-sizes 2,4,6,8 --modes compressed --profile coordinated

# Only map 3
python multi_test_driver.py --maps 3 --team-sizes 2,4,6,8

# Original DARE comparison (disables all added communication/collision/deadlock wrappers)
python multi_test_driver.py --maps all --map-count 100 --profile original_dare
```

- Results CSV is written to `{run_path}/multi_robot_outputs/run_{timestamp}_.../results.csv`
- Communication modes: `none` (no map transmitted), `raw` (full snapshot at every contact), `compressed` (per-peer incremental delta).

---

## 4. Reproducing the Chapter 4 Experiments (recommended)


```bash
# Stage 1: compare single-robot behaviour of DARE-L6 / LiteDARE-L4 / LiteDARE-L2 and auto-select the downstream model
python paper_experiments/run_chapter4.py --stage compare

# Stage 2: run multi-robot ablation + communication comparison with the selected LiteDARE checkpoint
python paper_experiments/run_chapter4.py --stage downstream

# Run both stages in one go
python paper_experiments/run_chapter4.py --stage all
```

Edit `paper_experiments/chapter4_config.py` before testing
- The three checkpoint paths `DARE_L6_CHECKPOINT` / `LITE_L4_CHECKPOINT` / `LITE_L2_CHECKPOINT` (set before run);
- `RANDOM_SEED=42`, `MAPS`, `MAP_COUNT`, `RUNS_PER_MAP`, `TEAM_SIZES`;
- Selection margins `DELTA_SUCCESS_RATE=0.02`, `DELTA_FINAL_COVERAGE=0.01`, `DELTA_COVERAGE_AUC=0.02`;
- `BOOTSTRAP_SAMPLES=5000`, `SELECTION_TIE_TOLERANCE=0.05`.

We build on the codebase from [DARE](https://github.com/marmotlab/DARE/tree/main).

---
