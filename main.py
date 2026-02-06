import os
import cv2
import joblib
import numpy as np

from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ======================
# Configuration
# ======================
DATA_ROOT = "./data/caltech101"
MODEL_PATH = "best_svm_hog_pca.joblib"
IMG_SIZE = (96, 96)
RANDOM_STATE = 42


# ======================
# Feature Extraction
# ======================
def extract_hog(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )


# ======================
# Dataset Statistics
# ======================
def dataset_statistics():
    print("\n------- Dataset Statistics -------")

    class_names = sorted([
        c for c in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, c))
    ])

    total_images = 0
    counts = {}

    min_area, max_area = float("inf"), 0
    min_info, max_info = None, None

    for cls in class_names:
        class_dir = os.path.join(DATA_ROOT, cls)
        images = os.listdir(class_dir)

        counts[cls] = len(images)
        total_images += len(images)

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            area = h * w

            if area > max_area:
                max_area = area
                max_info = (w, h, img_path)

            if area < min_area:
                min_area = area
                min_info = (w, h, img_path)

    print("Number of classes:", len(class_names))
    print("Total images:", total_images)

    print("Class with max images:", max(counts, key=counts.get), counts[max(counts, key=counts.get)])
    print("Class with min images:", min(counts, key=counts.get), counts[min(counts, key=counts.get)])

    print("Largest image :", f"{max_info[0]} x {max_info[1]}", "|", max_info[2])
    print("Smallest image:", f"{min_info[0]} x {min_info[1]}", "|", min_info[2])


# ======================
# Training
# ======================
def load_dataset():
    """Load dataset and extract features."""
    X, y = [], []

    class_names = sorted([
        c for c in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, c))
    ])

    print("Loading dataset ...")
    for cls in tqdm(class_names, desc="Classes"):
        class_dir = os.path.join(DATA_ROOT, cls)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            feat = extract_hog(img_path)

            if feat is not None:
                X.append(feat)
                y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y)


def train_model():
    print("\n------- Training Model -------")

    X, y = load_dataset()
    print("Feature shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(whiten=True)),
        ("svm", SVC(class_weight="balanced"))
    ])

    param_dist = [
    # # Linear SVM
    # {
    #     "pca__n_components": [50, 80, 100, 150],
    #     "svm__kernel": ["linear"],
    #     "svm__C": [0.1, 1, 5, 10]
    # },
    # RBF SVM
    {
        "pca__n_components": [0.90, 0.95, 0.98],
        "svm__kernel": ["rbf"],
        "svm__C": [ 1, 5, 10],
        "svm__gamma": [0.001, 0.01, 0.1]
    },
    # Polynomial SVM
    {
        "pca__n_components": [0.90, 0.95, 0.98],
        "svm__kernel": ["poly"],
        "svm__C": [ 1, 5, 10],
        "svm__degree": [2, 3],
        "svm__gamma": ["scale", "auto"],
        "svm__coef0": [0.0, 0.1, 0.5] 
    }
    ]

    print("Hyperparameter search ...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("Best params:", search.best_params_)
    joblib.dump(best_model, MODEL_PATH)
    print("✅ Model saved:", MODEL_PATH)

    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# ======================
# Prediction Helpers
# ======================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train it first.")
    return joblib.load(MODEL_PATH)


def predict_folder(model, folder_path):
    y_true, y_pred = [], []

    for img_name in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)
        if not os.path.isfile(img_path):
            continue

        feat = extract_hog(img_path)
        if feat is None:
            continue

        pred = model.predict(feat.reshape(1, -1))[0]
        true = os.path.basename(folder_path)

        y_true.append(true)
        y_pred.append(pred)

        if pred != true:
            print(f"❌ {img_name:30s} -> {pred}")

    return y_true, y_pred


# ======================
# Prediction Modes
# ======================
def predict_single_class():
    class_name = input("Enter class name: ").strip()
    folder_path = os.path.join(DATA_ROOT, class_name)

    if not os.path.isdir(folder_path):
        print("❌ Class folder not found")
        return

    model = load_model()
    print(f"\n📂 Predicting class: {class_name}")
    print("-" * 50)

    y_true, y_pred = predict_folder(model, folder_path)

    print("\n-------- Metrics --------")
    print(classification_report(
        y_true,
        y_pred,
        labels=[class_name],
        target_names=[class_name],
        zero_division=0
    ))


def predict_entire_dataset():
    model = load_model()

    y_true_all, y_pred_all = [], []

    for cls in sorted(os.listdir(DATA_ROOT)):
        folder_path = os.path.join(DATA_ROOT, cls)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n📂 Predicting class: {cls}")
        print("-" * 50)

        y_true, y_pred = predict_folder(model, folder_path)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    print("\n======== Overall Report ========")
    print(classification_report(y_true_all, y_pred_all, zero_division=0))

    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"Overall Accuracy: {acc:.4f}")


# ======================
# Menu
# ======================
def main():
    while True:
        print("\n====== MENU ======")
        print("1 - Dataset Statistics")
        print("2 - Train Model")
        print("3 - Predict One Class")
        print("4 - Predict Entire Dataset")
        print("0 - Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            dataset_statistics()
        elif choice == "2":
            train_model()
        elif choice == "3":
            predict_single_class()
        elif choice == "4":
            predict_entire_dataset()
        elif choice == "0":
            break
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
