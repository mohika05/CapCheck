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

## Robustness Benchmark & Automated Metrics
To run the model against the 14 real-world transformations (JPEG, Blur, Noise, Resizing, Jitter, Cropping):

```bash
python predict.py /path/to/image_directory --report-transforms
```

### Directory Structure for Accuracy Scoring
If you want the script to automatically calculate and print `Accuracy` and `ROC-AUC` metrics in the terminal, your images must be placed inside specific subfolders that indicate their true label. We recommend using the provided `data_test/validation` structure:

* **Real Images:** Place inside `data_test/validation/real/`
* **AI Images:** Place inside `data_test/validation/aigc/`

When you run `python predict.py data_test/validation`, the script will automatically calculate accuracy based on the `real/` and `aigc/` subfolders inside it.

*Note: If images are not in these recognized folders, the script will still successfully generate the JSON predictions, but accuracy metrics will be skipped.*

**Benchmark Output:**
1. Generates `predictions_local.json` for the clean images.
2. Prints a terminal scorecard evaluating Accuracy and ROC-AUC per condition.
3. Generates `results/robustness_local.json` containing the detailed metric breakdown.
