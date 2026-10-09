# Colon Cancer Detection from Histopathology Images

Binary classification of colon tissue images as **benign** or **adenocarcinoma**, using OpenCV preprocessing and a transfer-learning CNN (PyTorch).

> Originally built during my undergraduate studies. Rebuilt and extended in 2026 after the original codebase was lost.

## Problem

Colorectal cancer diagnosis relies on pathologists examining H&E stained tissue slides, which is slow and subject to inter-observer variability. This project trains a model to flag adenocarcinoma in colon tissue patches as a decision-support tool.

## Dataset

[LC25000](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) (Borkowski et al., 2019), colon subset:

| Class | Folder | Images |
|---|---|---|
| Benign colon tissue | `colon_n` | 5,000 |
| Colon adenocarcinoma | `colon_aca` | 5,000 |

Split: 70% train / 15% validation / 15% test, stratified, fixed seed.

**Known limitation:** LC25000 was produced by augmenting a smaller set of original images, so near-duplicate patches can land in different splits. Test scores on this dataset are optimistic compared with performance on images from a new hospital.

## Approach

1. **OpenCV preprocessing.** CLAHE on the lightness channel (LAB color space) to boost local contrast of nuclei and glands while keeping stain colors. Then resize to 224×224.
2. **Augmentation.** Random flips, 90° rotations, and mild brightness/contrast jitter to simulate staining variation. Tissue has no fixed orientation, so these are label-preserving.
3. **Model.** ImageNet-pretrained ResNet-18 (ResNet-50 and EfficientNet-B0 also supported) with a new dropout + linear head, fine-tuned end to end.
4. **Training.** AdamW, cosine learning rate schedule, cross-entropy loss. The best checkpoint is chosen on validation accuracy.
5. **Evaluation.** Accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, confusion matrix, and ROC curve on the untouched test split.

## Results

Test set (n = _fill in_), ResNet-18 with CLAHE:

| Metric | Score |
|---|---|
| Accuracy | _fill in from outputs/metrics.json_ |
| Precision | |
| Recall (sensitivity) | |
| Specificity | |
| F1 | |
| ROC-AUC | |

<!-- After running evaluation, add: ![Confusion matrix](outputs/confusion_matrix.png) -->

## Project structure

```
colon-cancer-detection/
├── app.py                 # Streamlit demo
├── scripts/
│   └── split_data.py      # Stratified train/val/test split CSV
├── src/
│   ├── data.py            # OpenCV preprocessing, augmentation, Dataset
│   ├── model.py           # Transfer-learning backbones
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Test metrics, confusion matrix, ROC curve
│   ├── predict.py         # Single-image inference
│   └── utils.py
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. Download LC25000 from Kaggle and unzip into data/
#    (expects data/lung_colon_image_set/colon_image_sets/{colon_aca,colon_n})

# 2. Create the split
python scripts/split_data.py --src data/lung_colon_image_set/colon_image_sets --out data/splits.csv

# 3. Train (a GPU helps; Google Colab or Kaggle notebooks work well)
python -m src.train --csv data/splits.csv --arch resnet18 --epochs 10

# 4. Evaluate on the test split
python -m src.evaluate --csv data/splits.csv --ckpt outputs/best_model.pt

# 5. Predict one image, or launch the demo
python -m src.predict path/to/image.jpeg
streamlit run app.py
```

To measure what CLAHE contributes, train once with `--no-clahe` and compare.

## Disclaimer

Research and educational project. Not a medical device and not validated for clinical use.
