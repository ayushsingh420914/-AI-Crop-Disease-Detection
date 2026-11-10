# AI Yield: Crop Disease Detection and Management

A runnable Python project that classifies crop-leaf images using a MobileNetV3-Small image model and displays general, cautious crop-management suggestions in a Streamlit web app.

## What this project does

1. Trains an image classifier using a folder-based crop-leaf dataset.
2. Splits each class into training and validation images.
3. Saves the best model and the matching class labels.
4. Accepts a leaf image through a local web page.
5. Displays the top prediction, model confidence score, the next two predictions, and general management suggestions.
6. Also supports prediction from the Windows/Linux command line.

**Important:** This version detects image classes. It does not forecast crop yield in kilograms or tonnes. The prediction is a screening aid, not a confirmed diagnosis.

## 1. Requirements

- Python 3.10 or 3.11 is recommended.
- Internet is needed to install packages and, by default, to download the optional ImageNet starting weights.
- Training is faster with an NVIDIA GPU, but CPU training is supported.

## 2. Create and activate a virtual environment (Windows)

Open Command Prompt or PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not available, install Python 3.10/3.11 and use `python -m venv .venv`.

## 3. Download the dataset

This project uses the **PlantVillage** leaf-image dataset as a starter dataset. Download it from:

https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

On Kaggle, sign in, select **Download**, extract the ZIP, and place the extracted class folders somewhere inside `data/raw/`.

The expected structure is a folder per class, for example:

```text
data/
└── raw/
    └── PlantVillage/
        ├── Apple___Apple_scab/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── Apple___Black_rot/
        ├── Apple___healthy/
        ├── Tomato___Early_blight/
        └── Tomato___healthy/
```

The exact class list depends on the dataset version you download. The training script searches inside `data/raw/` and identifies the folder whose immediate subfolders contain images, so an extra extracted parent folder is okay.

The dataset page lists a **CC BY-NC-SA 4.0** licence. Review its conditions before redistributing the dataset or using it commercially.

### Optional Kaggle CLI download

You can also install the Kaggle CLI and configure your Kaggle API token, then run:

```powershell
python -m pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
```

If the command extracts files into an unexpected nested folder, the training script should still locate the class directory automatically. If it does not, pass the exact extracted folder to `--data_dir`.

## 4. Train the model

From the project root, with the virtual environment active:

```powershell
python train.py --data_dir data/raw --epochs 12 --batch_size 32
```

The script automatically:

- finds the image-class directory;
- creates a stratified 80/20 train/validation split;
- applies training-time image augmentation;
- trains MobileNetV3-Small;
- saves the best validation model and class mapping into `models/`.

Expected saved files:

```text
models/
├── crop_disease_model.pth
├── classes.json
├── model_config.json
└── training_history.json
```

### Useful training options

```powershell
# Faster/smaller run for a computer with limited memory
python train.py --data_dir data/raw --epochs 5 --batch_size 16 --image_size 160

# Force CPU even if a GPU is available
python train.py --data_dir data/raw --epochs 12 --cpu

# Do not use/download ImageNet starting weights (fully offline model initialization)
python train.py --data_dir data/raw --epochs 12 --no-pretrained
```

If the pretrained weights cannot be downloaded, the script automatically falls back to random initialization and continues. Training from random initialization may need more epochs and can produce weaker results.

## 5. Run the web application

After training finishes:

```powershell
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`. Open it in your browser, upload a clear leaf photo, and view the result.

## 6. Predict from the command line

```powershell
python predict.py "path\to\leaf_photo.jpg"
```

For example:

```powershell
python predict.py "data\sample_leaf.jpg"
```

## 7. Common problems

**`Could not find a dataset folder`**  
Make sure you extracted the dataset. Each disease/healthy class should be a separate folder containing image files. Pass that folder or its parent using `--data_dir`.

**`Each class needs at least 2 images`**  
At least two images are required in every class to create both training and validation examples. Remove empty/one-image folders or use a better dataset copy.

**Training is too slow**  
Use `--batch_size 16 --image_size 160 --epochs 5`, close other heavy applications, or train on a GPU-enabled machine. These settings are for a quick test, not a final performance claim.

**`torch` / `torchvision` installation error**  
Use Python 3.10 or 3.11 in a fresh virtual environment and install the matching PyTorch and Torchvision builds using the official PyTorch installation selector: https://pytorch.org/get-started/locally/ . Then run `pip install -r requirements.txt` again.

## 8. Project files

- `train.py` — dataset discovery, split, training, validation, and model saving.
- `model.py` — MobileNetV3-Small model definition.
- `predictor.py` — shared model loading and image prediction functions.
- `predict.py` — command-line prediction.
- `app.py` — Streamlit user interface.
- `recommendations.py` — general management suggestions based on predicted class names.
- `requirements.txt` — Python dependencies.

## 9. Academic/report limitation to mention

PlantVillage is a controlled leaf-image dataset. Results on clean dataset images do not guarantee the same performance on real farm photos, where lighting, backgrounds, camera quality, multiple diseases, and local crop varieties can differ. For a stronger field-ready project, collect and evaluate additional locally captured images (for example, rice, wheat, maize, or potato leaves from the target region) and report a separate test-set evaluation.
