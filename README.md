# 🖼️ Caltech101 Image Classification  
### Classical Computer Vision with HOG + PCA + SVM

A **clean, reproducible, and well-structured** classical machine learning pipeline for multi-class image classification on the **Caltech101** dataset.

This project intentionally avoids deep learning to demonstrate **strong fundamentals** in feature engineering, dimensionality reduction, and model evaluation.

---

## ✨ Highlights

- 🧠 Histogram of Oriented Gradients (HOG) feature extraction  
- 📉 Dimensionality reduction using Principal Component Analysis (PCA)  
- ⚙️ Support Vector Machines (RBF & Polynomial kernels)  
- 🎯 Hyperparameter tuning with `RandomizedSearchCV`  
- ⚖️ Macro F1-score optimization for class imbalance  
- 🔁 Fully reproducible experiments  
- 🧩 Modular, readable, and production-style code  

---

## 📸 Dataset

**Caltech101**

- ~101 object categories  
- Highly imbalanced class distribution  
- Varying image resolutions  

Expected directory structure:

```
data/
└── caltech101/
    ├── accordion/
    ├── airplane/
    ├── anchor/
    └── ...
```


---

## 🗂️ Project Structure

```
.
├── main.py
├── best_svm_hog_pca.joblib   # generated after training
├── data/
│   └── caltech101/
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Requirements
- Python 3.8+

### Dependencies

```bash
pip install numpy opencv-python scikit-image scikit-learn tqdm joblib
```

---

## 🚀 Usage

Run the application:

```bash
python main.py
```

You will be presented with an interactive menu:

```
1 - Dataset Statistics
2 - Train Model
3 - Predict One Class
4 - Predict Entire Dataset
0 - Exit
```

---

## 🧪 Menu Options Explained

### Dataset Statistics
- Number of classes
- Total images
- Most / least populated classes
- Largest & smallest image resolutions

### Train Model
- Extracts HOG features from all images
- Stratified train / test split
- PCA + SVM pipeline
- Hyperparameter tuning with 5-fold CV
- Optimizes **macro F1-score**
- Saves the best model to disk

### Predict One Class
- Evaluates all images from a single class folder
- Prints misclassified samples
- Outputs precision, recall, and F1-score for that class

### Predict Entire Dataset
- Runs inference over the full dataset
- Aggregates predictions across all classes
- Prints overall classification report and accuracy  

> ⚠️ Includes training samples. For exploratory analysis only.

---

## 🧠 Model Architecture

**Preprocessing**
- Resize images to 96 × 96
- Convert to grayscale

**HOG**
- orientations: 9
- pixels_per_cell: (8, 8)
- cells_per_block: (2, 2)
- block_norm: L2-Hys

**Classifier**
- SVM (RBF & Polynomial)
- class_weight: balanced

---

## 📊 Evaluation

Metrics:
- Accuracy
- Precision
- Recall
- F1-score (macro)

Macro F1-score is used to properly handle class imbalance.
