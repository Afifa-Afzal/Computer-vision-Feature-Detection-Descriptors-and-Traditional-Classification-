# Feature Detection, Descriptors, and Traditional Classification

 implemented in Google Colab using OpenCV and scikit-learn.

## Overview

This project covers a classical (non-deep-learning) computer vision pipeline:

1. **Edge detection** with Sobel and Canny, with an explanation of the results
2. **Keypoint detection and descriptors** using SIFT and ORB
3. **Descriptor matching** between two related images, with visualized matches
4. **Classification** of extracted features using SVM and k-NN
5. **Evaluation** with accuracy, precision, recall, F1-score, and confusion matrices
6. **Report** (PDF) summarizing the method and results

## Methods

| Task | Technique | Details |
|---|---|---|
| Edge detection | Gaussian blur + Sobel, Canny | Canny thresholds 100 / 200 |
| Keypoints | SIFT, ORB | SIFT: 128-d float descriptors, ORB: 32-byte binary descriptors |
| Matching | BFMatcher + Lowe's ratio test (0.75) | L2 distance for SIFT, Hamming for ORB; RANSAC homography to remove outliers |
| Features for classification | Bag of Visual Words | Dense SIFT on a 8x8 grid, K-Means vocabulary (K=30), normalized histograms |
| Classifiers | SVM (RBF kernel, C=10), k-NN (k=3) | Features standardized with `StandardScaler` |
| Metrics | scikit-learn | Accuracy, precision, recall, F1-score, confusion matrix |

## Dataset

The classification task uses the **Olivetti Faces** dataset (loaded through `sklearn.datasets.fetch_olivetti_faces`), with 5 people (classes) and 10 images per person. Images are resized to 128x128 and split 70% train / 30% test (stratified, `random_state=42`).

For edge detection, keypoints, and matching, personal images were used (see the notebook).

> The test set is small, so a single wrong prediction changes the accuracy noticeably.

## Repository Structure

```
.
├── notebook.ipynb              # Full Colab notebook (all tasks)
├── Assignment_Report.pdf       # 3-5 page report
├── README.md
└── outputs/
    ├── edges_sobel.png
    ├── edges_canny.png
    ├── keypoints_sift.png
    ├── keypoints_orb.png
    ├── matches_sift.png
    ├── matches_orb.png
    ├── matches_sift_ransac.png
    ├── confusion_matrices.png
    └── metrics.txt
```

## How to Run

### Option 1: Google Colab (recommended)

1. Open `notebook.ipynb` in Google Colab (File > Upload notebook, or use the GitHub tab).
2. Run all cells from top to bottom (Runtime > Run all).
3. Upload your own images when the upload cells ask for them.


Note: `opencv-python` 4.4 or later is required for `cv2.SIFT_create()`. The `google.colab` upload cells only work in Colab; replace them with local file paths when running locally.

## Key Observations

- Canny produces thinner and cleaner edges than Sobel because of non-maximum suppression and hysteresis thresholding.
- SIFT is more robust to scale and rotation changes, while ORB is much faster.
- Lowe's ratio test and RANSAC remove most incorrect matches.


## Technologies

- Python 3
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- ReportLab (PDF generation)
- Google Colab

