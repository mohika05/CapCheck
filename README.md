# AI-Generated Image Detection Inference Pipeline

This repository contains the evaluation script `predict.py` for scoring images as Authentic (Real) or AI-Generated.

## Standard Evaluation
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

## Robustness Benchmark
To run the model against the 14 real-world transformations (JPEG, Blur, Noise, Resizing, Jitter, Cropping):

```bash
python predict.py /path/to/image_directory --report-transforms
```

**Output:**
1. Generates `predictions_local.json` for the clean images.
2. Prints a terminal scorecard evaluating Accuracy and ROC-AUC per condition.
3. Generates `results/robustness_local.json` containing the detailed metric breakdown.
