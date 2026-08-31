# Track 5: Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 â€” Track 5 Submission

## Project Overview

This project detects AI-generated images under real-world post-processing conditions â€” JPEG compression, blurring, resizing, noise, colour jitter, and cropping. The detector combines two complementary feature streams into a lightweight classifier:

**CLIP ViT-L/14 (768-D)** â€” a frozen Vision Transformer with 14Ã—14 patches that captures semantic and structural features. The finer patch resolution (vs ViT-B/32's 32Ã—32) is deliberate: DALL-E Advanced artifacts tend to be localised â€” subtle boundary inconsistencies, texture anomalies, hand distortions â€” and coarser patches average these signals away.

**DCT frequency features (64-D)** â€” the top-left 8Ã—8 block of the Discrete Cosine Transform, normalised and log-compressed. AI generators leave characteristic frequency-domain fingerprints in low-frequency coefficients that differ from real camera images.

The concatenated 832-D vector passes through a 3-layer MLP classifier (~500K parameters) trained with binary cross-entropy loss and augmentation-aware exposure to all 14 Track 5 transforms.

### Results on WildFake (13,841 images)

| Metric | Value |
|---|---|
| Clean Accuracy | 0.8102 |
| ROC-AUC | 0.9056 |
| Mean Accuracy (all 15 conditions) | 0.7491 |
| AIGC Precision | 0.9283 |
| AIGC Recall | 0.7617 |
| Worst Condition | Blur Ïƒ=2.0 (0.5937) |

---

## Setup and Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended for feature extraction and training
- CPU is sufficient for inference only

### Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`: `torch`, `open-clip-torch`, `Pillow`, `numpy`, `scikit-learn`, `opencv-python-headless`, `datasets`, `huggingface-hub`

### Repository structure

```
track5/
â”œâ”€â”€ config.py                    # Unified config â€” auto-switches between local and GPU
â”œâ”€â”€ predict.py                   # Inference script (competition deliverable)
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ stream_extract.py        # Feature extraction (CLIP + DCT + augmentations)
â”‚   â””â”€â”€ train.py                 # MLP classifier training
â”œâ”€â”€ jobs/
â”‚   â”œâ”€â”€ job_extract.sh           # SLURM script for feature extraction
â”‚   â”œâ”€â”€ job_train.sh             # SLURM script for training
â”‚   â””â”€â”€ job_predict.sh           # SLURM script for WildFake evaluation
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ classifier.pt            # Trained model checkpoint
â”‚   â”œâ”€â”€ predictions.json         # WildFake clean predictions
â”‚   â”œâ”€â”€ predictions_local.json   # Local test predictions
â”‚   â””â”€â”€ robustness_summary.json  # Per-transform metrics
â””â”€â”€ RUNNING_INSTRUCTIONS.md      # Quick-start inference guide
```

### Environment switching

The codebase uses a single `config.py` that auto-selects paths based on the `ENV` environment variable:

- **Local (default):** `ENV` is unset â†’ uses small test datasets and local paths
- **GPU cluster:** `export ENV=tc1` â†’ uses full datasets and TC1 cluster paths (set automatically in SLURM job scripts)

No manual config switching is needed.

---

## Steps to Reproduce

### Quick Start: Inference Only

**Standard Evaluation**  
To score a directory of images and output a JSON list of predictions:

```bash
python predict.py /path/to/image_directory
```
*(Alternative flag: `--input_dir /path/to/image_directory`)*

**Output:** 
Generates `results/predictions_local.json` with the following schema:
```json
[
  {
    "image_path": "/path/to/image_directory/image1.jpg",
    "pred": 0.042
  },
  {
    "image_path": "/path/to/image_directory/image2.png",
    "pred": 0.961
  }
]
```
*(0.0 = Authentic, 1.0 = AI-Generated)*

---

**Robustness Benchmark & Automated Metrics**  
To run the model against the 14 real-world transformations (JPEG, Blur, Noise, Resizing, Jitter, Cropping):

```bash
python predict.py /path/to/image_directory --report-transforms
```

**Directory Structure for Accuracy Scoring**  
If you want the script to automatically calculate and print `Accuracy` and `ROC-AUC` metrics in the terminal, your images must be placed inside specific subfolders that indicate their true label. We recommend using the provided `input` structure:

* **Real Images:** Place inside `input/real/`
* **AI Images:** Place inside `input/ai/`

When you run `python predict.py input`, the script will automatically calculate accuracy based on the `real/` and `ai/` subfolders inside it.

*Note: If images are not in these recognized folders, the script will still successfully generate the JSON predictions, but accuracy metrics will be skipped.*

**Benchmark Output:**
1. Generates `predictions_local.json` for the clean images.
2. Prints a terminal scorecard evaluating Accuracy and ROC-AUC per condition.
3. Generates `results/robustness_local.json` containing the detailed metric breakdown.

### Full Reproduction: Extract â†’ Train â†’ Predict

#### 1. Prepare data

| Dataset | Source | Download |
|---|---|---|
| CIFAKE | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | Place under `data/cifake/train/` and `data/cifake/test/` |
| SynthBuster | [Zenodo](https://zenodo.org/records/10066460) | Place under `data/synthbuster/` |
| WildFake (eval only) | [ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake) | Place under `data/validation/` |

SID and Tiny-GenImage are streamed directly from HuggingFace during extraction â€” no manual download needed.

#### 2. Extract features (~2.7 hours on V100)

```bash
# On SLURM cluster:
sbatch jobs/job_extract.sh

# Or locally (much slower, CPU):
python src/stream_extract.py --overwrite
```

This processes all five training sources in a single pass, applying CLIP ViT-L/14 encoding + DCT extraction for each image's clean and augmented versions. Output: 574,000 feature vectors across 29 compressed NumPy shards.

#### 3. Train classifier (~9 minutes on V100)

```bash
sbatch jobs/job_train.sh
# or:
python src/train.py
```

Loads all shards, undersamples to 50/50 class balance, fits z-score normalisation, and trains the MLP for 75 epochs. Best checkpoint is saved to `results/classifier.pt` based on internal validation accuracy (CIFAKE test, 20K images).

#### 4. Evaluate on WildFake

```bash
sbatch jobs/job_predict.sh
# or:
python predict.py /path/to/validation --report-transforms
```

Runs clean inference + all 14 transforms. Outputs `predictions.json` and `robustness_summary.json`.

#### 5. Verify output format

```bash
python -c "
import json
data = json.load(open('results/predictions.json'))
print(f'Entries: {len(data)}')
print(f'Sample: {data[0]}')
"
```

Expected: 13,841 entries, each with `image_path` (string) and `pred` (float, 0.0â€“1.0).

---

## Limitations and What We Would Improve

### Current Limitations

**Generator coverage gap** â€” the model was trained on DALL-E 2/3 (via SynthBuster) but evaluated on DALL-E Advanced. These are closely related but not identical. In informal testing, images from generators the model has never seen at all â€” particularly Gemini (Imagen 3) â€” score near zero despite being AI-generated. The model detects generators it has learned about, but does not fully generalise to entirely unseen architectures.

**Heavy degradation failure** â€” the two worst conditions are blur Ïƒ=2.0 (59.4%) and resize 0.25Ã— (62.5%). Both destroy the high-frequency information that the DCT branch relies on, and the aggressive spatial distortion also degrades CLIP's patch-level features. These transforms genuinely destroy the signal rather than merely obscuring it.

**Text overlay and screenshot robustness** â€” DALL-E Advanced images with text overlays (e.g. aesthetic Pinterest-style captions) or images screenshotted from browsers tend to fool the model. Screenshots strip frequency artifacts through re-encoding, and text shifts CLIP's semantic interpretation away from the photographic space it was trained on.

**Internal validation mismatch** â€” checkpoint selection uses CIFAKE test (SD 1.4, 32Ã—32), which has a very different distribution from WildFake (DALL-E Advanced, full resolution). The optimal threshold found on internal validation (0.9) does not transfer to WildFake. A DALL-E-family validation set would improve model selection.

### What We Would Improve Given More Time

**Patch-level attention pooling** â€” instead of relying on CLIP's global mean-pooled embedding, attend directly to ViT-L/14's local token embeddings to explicitly localise artifacts rather than hoping the global representation captures them.

**Adaptive DCT gating** â€” a learned gate that downweights DCT features when the image shows signs of heavy degradation (strong blur, aggressive downscaling), preventing the DCT branch from actively hurting performance under conditions that destroy frequency information.

**Broader generator diversity** â€” integrate the Community Forensics and AntiFake datasets to cover Imagen (Google), Flux (Black Forest Labs), and other emerging generator families the model has not yet seen.

**Threshold calibration** â€” tune the decision threshold on a small held-out DALL-E-family validation set, separate from both the training data and the WildFake evaluation set, to improve the accuracy-recall balance on the target distribution.

**Frequency-domain augmentation** â€” apply random spectral perturbations during training to prevent the DCT branch from overfitting to any single generator's frequency signature.

---

## Team Member Contributions

<!-- Replace Person A/B/C/D with actual names -->

| Member | Role | Contributions |
|---|---|---|
| **Person A** | Data Pipeline & Infrastructure | TC1 GPU cluster setup and access, dataset downloads and organisation on the cluster, data directory structure |
| **Person B (Mohika)** | Model Development | Feature extraction pipeline (`stream_extract.py`), training pipeline (`train.py`), CLIP + DCT architecture design, domain gap diagnosis and dataset selection (SynthBuster, Tiny-GenImage, ViT-L/14 upgrade), augmentation implementations, SLURM job scripts, `config.py` environment switching |
| **Person C** | Inference & Evaluation | Inference script (`predict.py`), robustness evaluation framework, prediction output formatting |
| **Person D** | Documentation & Presentation | README, Devpost write-up, demo video, error analysis documentation |

---

## Robustness Evaluation Summary

| Transform | Accuracy | ROC-AUC | Real Recall | AIGC Recall |
|---|---|---|---|---|
| **Clean** | **0.8102** | **0.9056** | 0.8960 | 0.7617 |
| JPEG q=90 | 0.8140 | 0.9054 | 0.8976 | 0.7668 |
| JPEG q=70 | 0.7929 | 0.8770 | 0.8776 | 0.7450 |
| JPEG q=50 | 0.7611 | 0.8357 | 0.8161 | 0.7300 |
| JPEG q=30 | 0.7456 | 0.8316 | 0.8303 | 0.6977 |
| Blur Ïƒ=0.5 | 0.8281 | 0.9290 | 0.9504 | 0.7590 |
| Blur Ïƒ=1.0 | 0.7942 | 0.8734 | 0.8790 | 0.7462 |
| Blur Ïƒ=2.0 | 0.5937 | 0.5531 | 0.3741 | 0.7179 |
| Resize 0.5Ã— | 0.7857 | 0.8626 | 0.8649 | 0.7409 |
| Resize 0.25Ã— | 0.6247 | 0.6203 | 0.4700 | 0.7121 |
| Noise Ïƒ=0.02 | 0.7549 | 0.9035 | 0.9452 | 0.6473 |
| Noise Ïƒ=0.05 | 0.7255 | 0.8601 | 0.9200 | 0.6155 |
| Noise Ïƒ=0.10 | 0.6729 | 0.7695 | 0.8041 | 0.5988 |
| Jitter Â±0.2 | 0.7997 | 0.9001 | 0.9056 | 0.7399 |
| Crop 80% | 0.7327 | 0.9246 | 0.9804 | 0.5927 |

Overall accuracy (clean + all transforms): **0.7491** | Worst condition: Blur Ïƒ=2.0 (0.5937)

The model is particularly robust to JPEG compression â€” the most common real-world degradation â€” holding above 74% accuracy even at quality 30. Mild blur (Ïƒ=0.5) actually *improves* accuracy to 82.8%, likely because it suppresses high-frequency noise that sometimes triggers false positives.

---

## Error Analysis

### False Negatives (2,107 DALL-E Advanced fakes missed on clean)

The model misses roughly 24% of AI-generated images. These are not random failures â€” they concentrate in smooth, photorealistic images of natural scenes (landscapes, food, architecture) where DALL-E Advanced's generation quality is highest. The model's decision boundary was shaped primarily by SD 1.4 and DALL-E 2/3 training examples, and DALL-E Advanced's cleaner frequency signature sits closer to the real distribution than its predecessors.

Informal testing with fresh ChatGPT-generated images (September 2026) revealed a related pattern: images with text overlays, stylised layouts, or screenshot-style borders consistently score near zero. The re-encoding pipeline strips frequency artifacts the DCT branch relies on.

### False Positives (520 real images wrongly flagged)

AIGC precision is 92.8% â€” only 3.8% of real images are incorrectly flagged. The false positives cluster around real photographs with unusual processing: heavy HDR tone mapping, artistic filters, aggressive sharpening, or images captured through reflective surfaces. These processing steps create frequency patterns that resemble AI generation artifacts.

### Key Trade-off

The model operates at threshold 0.5, producing high real recall (89.6%) at the cost of lower AIGC recall (76.2%). The DCT branch provides strong signal on clean and mildly degraded images but becomes unreliable under heavy blur (Ïƒâ‰¥2.0) and aggressive downscaling (0.25Ã—), which destroy the frequency information it depends on. An adaptive gating mechanism that downweights DCT under detected degradation would mitigate this.
