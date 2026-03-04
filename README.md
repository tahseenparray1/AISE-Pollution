# Baseline Pipeline with FNO2D Model

This repository provides a **reference baseline pipeline** for the Kaggle competition **[AISEHACK Theme 2 - Pollution Forecasting](https://www.kaggle.com/competitions/aisehack-theme-2/overview)**, demonstrating a complete end-to-end workflow using a **2D Fourier Neural Operator (FNO2D)** implementation. It is intended as a **clear, minimal, and reproducible example** rather than a production-optimized solution.

---

## Purpose of This Repository

- Serve as a **starter template** for building Kaggle notebook submissions  
- Demonstrate how to structure:
  - Dataset preparation
  - Model training
  - Inference and submission generation
- Reproduce the **Baseline Benchmark** currently listed on the leaderboard
- Provide a clean reference that users can freely modify or extend

- This repository is **not restrictive**. 

---

## Pipeline Overview

The workflow is organized into **three sequential stages**:

1. **Dataset Preparation(prepare_dataset.py)**  
   Converts raw competition data into normalized, fixed-length spatio-temporal samples suitable for model training.

2. **Model Training(train.py)**  
   Trains an FNO2D model using the prepared datasets to forecast future PM2.5 concentration fields.

3. **Inference(infer.py)**  
   Runs the trained model on unseen test data and generates predictions in the format required for Kaggle submission.

- Each stage produces artifacts that are consumed by the next stage.

- Configurations of each script is given in their corresponding yaml files inside `configs/`

---

## Baseline Benchmark

- The training configuration in this repository **matches the official baseline benchmark** shown on the Kaggle leaderboard.
- All hyperparameters, features, and preprocessing steps are chosen to replicate that baseline faithfully.


---

## Using This Repository with Kaggle

- This repository is designed to be used **as a Kaggle dataset**, not run directly on local machines.

### Recommended Setup

1. Upload this repository as a **Kaggle Dataset**
2. Open the provided **baseline Kaggle notebook**: **[Link](https://www.kaggle.com/code/siddharthandileep/baseline-run-aisehack-test/)**, Copy and Edit the same.
3. Add:
   - This repository (as a dataset)
   - The official competition dataset
4. Set the accelerator to **GPU (P100)**
5. Click **Save & Run All**

The execution will emulate the baseline run:
- Read scripts and configs from this repository
- Execute dataset preparation, training, and inference
- Produce the final `preds.npy` submission file

---

## Intended Audience

This codebase is suitable for:
- First-time participants looking for a **clean starting point**
- Users unfamiliar with FNO-based spatio-temporal modeling
- Competitors who want a **working end-to-end baseline** before experimentation

---

## Extra Notes

- This repository emphasizes **clarity and reproducibility** over aggressive optimization, using memory-mapped NumPy arrays to operate reliably within Kaggle memory constraints. All scripts are fully configurable via YAML files, enabling easy and controlled experimentation.

- **Performance note:** the current training dataloader builds each sample by reading feature arrays individually and stacking them on-the-fly. While this keeps memory usage low and the implementation easy to follow, it introduces a **feature-wise I/O bottleneck** that can significantly increase training time—especially when training for many epochs, using larger batch sizes, or scaling to wider and deeper models. Participants planning longer training runs or higher-capacity models are **strongly encouraged to optimize this step**.

---


## Kaggle Data Locations for the Pipeline

- **Raw training data:** `/kaggle/input/competitions/aisehack-theme-2/raw/<MONTH>/<feature>.npy`
- **Test inputs:** `/kaggle/input/competitions/aisehack-theme-2/test_in/<feature>.npy`
- **Min–max statistics:** `/kaggle/input/competitions/aisehack-theme-2/stats/feat_min_max.mat`
- **Prepared datasets:** `/kaggle/temp/data/train/`, `/kaggle/temp/data/val/`
- **Checkpoints & logs:** `/kaggle/working/experiments/baseline/`
- **Final predictions:** `/kaggle/working/preds.npy`

---

## Points to remember regarding working with Kaggle

- Using **GPU P100** accelerator you have a peak CPU RAM limit of 29GB.
- You can save data upto 20GB in `/kaggle/working/`, data stored here is accessible even after session is closed. So anything relevant to be saved like final test predictions have to be saved here. Rest you can save in `/kaggle/temp/` or any directory outside of `/kaggle/working/`, they won't be saved once session is over.
- Keep in mind of all other logistical constraints imposed by Kaggle like session time limit, available weekly GPU compute etc and make sure your pipeline is as robust as possible to these constraints.
- Anything inside `/kaggle/input/`is read-only.
- Make sure to save your final test predictions in `/kaggle/working/preds.npy` for evaluation in this competition.

---

## Dataset Preparation (`prepare_dataset.py`)
- Raw feature arrays of shape `(T, H, W)` are loaded month-wise, normalized using precomputed min–max statistics, and converted into fixed-length time-series samples using a sliding window (`horizon=26`, `stride=1`). 

- Wind components (`u10`, `v10`) are scaled to `[-1, 1]`, emission variables are clipped to `[0, 1]`, and all samples are randomly split into training (80%) and validation (20%) sets. 

- Samples from all configured months are concatenated feature-wise and saved as separate NumPy arrays for each feature.

**Outputs:**  
`/kaggle/temp/data/train/train_<feature>.npy`  
`/kaggle/temp/data/val/val_<feature>.npy`  
Each file has shape `(N, 26, H, W)`.

---

## Model Training (`train.py`)

- The training script loads the prepared datasets using memory-mapped NumPy arrays, stacks all meteorological and emission variables channel-wise, and trains an **FNO2D** model. 

- The first 10 timesteps are used as input to predict the next 16 timesteps, with supervision applied only on future `cpm25`. Training uses Adam optimization with L2 loss, a step learning-rate scheduler, periodic model checkpointing, and JSON-based logging.

**Outputs:**  
- Model checkpoints saved in : `/kaggle/working/experiments/baseline/checkpoints/fno_baseline_ep<epoch>.pt`  
- Training logs saved in : 
`/kaggle/working/experiments/baseline/logs/log.json`

---

## Inference (`infer.py`)

- Inference loads feature-wise test inputs, applies the same normalization as training, and feeds the first 10 timesteps into the trained FNO model to predict 16 future cpm25 timesteps. 

- Predictions are de-normalized back to physical units using stored min–max statistics and saved as a single NumPy file - `preds.npy`.

**Outputs:**  
`/kaggle/working/preds.npy` with shape `(N_test, H, W, 16)`.

---

## Execution Order
Run the pipeline sequentially:
`python prepare_dataset.py` → `python train.py` → `python infer.py`

Each stage depends strictly on outputs produced by the previous step.


## Baseline Notebook — Execution Details

**Hardware**
- GPU: P100

**Runtime**
- ~350 seconds per epoch
- ~9.8 hours for the complete run

**Disk Usage**
- ~78 GB written during execution
  - ~76 GB: Prepared train/val dataset stored at `/kaggle/temp/data` in float32

**Memory Usage**
- Peak RAM: < 12 GB / 30 GB
- Peak GPU Memory: < 4 GB / 16 GB

