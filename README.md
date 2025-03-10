# DermaVision AI

This project is designed as a tool to assist users in identifying skin diseases by analyzing uploaded images. Upon submitting a picture, the system provides feedback on the diagnosed condition and, in cases where the disease is classified as mpox, also evaluate its stages level.

## Table of Contents

1. [Demo](#demo)
2. [Installation](#installation)
3. [Reproducing this project](#repro)
4. [Reference](#reference)

<a name="demo"></a>

## 1. Example demo

Main Features:

- Image upload and preview functionality
- Diagnosis results of skin conditions
- Information about the skin condition
- Educational resources about the skin condition
- If the diagnosed disease is Monkeypox, details about the current stage of the infection will be provided.

### What to find where

File Structure:

```bash
repository
├── static/                          # Frontend assets
│ ├── style.css                      # CSS for styling
│ ├── script.js                      # JavaScript for image handling
│ └── about.js                       # JavaScript for the about page
├── templates/                       # HTML templates
│ ├── index.html                     # Main diagnosis page
│ └── about.html                     # About Page for Disease information and team details
├── src/                             # Backend source code
│ └── mpox/                          # Part 1: Skin Disease Classification
│   │   ├── app.py                   # Flask server and routing
│   │   ├── train.py                 # Training the model
│   │   ├── test.py                  # Testing the model
│   │   ├── output.py                # Prediction output
│ └── stages/                        # Part 2: Stages Classification
│   │   ├── augmented_data           # Data output from keras_augmentation.py
│   │   │  ├── ...                   # Files storing images, each categorized into their class
│   │   ├── classifiedData           # Files containing classified model data
│   │   │  ├── ...                   # Images categorized into their class
│   │   ├── draft                    # Code used for research purposes in the past but is currently not in use
│   │   │  ├── ...                   # Files containing different unused algorithms
│   │   ├── test_data                # A placeholder for testing a model with input images.
│   │   │  ├── ...                   # Files containing different unused algorithms
│   │   ├── 1_keras_augmentation.py  # Augmenting Limited Data
│   │   ├── 2_pca.py                 # Feature extraction using PCA
│   │   ├── 3_1_train_pca_rf.py      # Training the model
│   │   ├── 3_2_test_pca_rf.py       # Testing the model
│ └── resources/                     # Resource files and test data
└── README.md                        # Project documentation
└── requirements.txt                 # Requirements installation file
```

<a name="installation"></a>

## 2 Installation

Note: Make sure you have python and pip installed.

### 1. Clone the repository

```bash
git clone git@github.com:ryanmitchellin/DermaVision-AI.git
```

### 2. Set Up environment

Install the required Python dependencies:

```bash
cd 2024_3_project_08
pip install -r requirements.txt
```

### 3. Download the dataset

- Go to the source https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20 (Paul, J.).
- Download the dataset by executing the provided code in a new standalone .py file. (Note: Make sure you don't download the dataset as zip as it messes up the pathway of the inner files).

Alternatively, you can copy the following code and creates a separate python file (in this case: import.py), paste the code, and run `python ./import.py` in the parent folder of the cloned file.

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")

print("Path to dataset files:", path)
```

- The folder's path will be displayed in the terminal. Look for the folder named 'mpox-skin-lesion-dataset-version-20-msld-v20/versions/4' and relocate it to './src/mpox/.'
- Ensure that the file path './src/mpox/mpox-skin-lesion-dataset-version-20-msld-v20/versions/4' exists.
- Download the classified dataset. Original data is retrieved from the same dataset as above (Paul, J.). https://drive.google.com/file/d/1nbV4X2f4PvFahuJbwLZc9pmFaODSSN7j/view?usp=sharing
- Extract and relocate the file to './src/stages/'
- Ensure that the file path './src/stages/classifiedData' exists.

### 4. Augment data

```bash
python ./src/stages/1_keras_augmentation.py
```

### 5. Extract features from data using Principal Component Analysis (PCA).

```bash
python ./src/stages/2_pca.py
```

### 6. Train the first model

```bash
python ./src/mpox/train.py
```

#### key features:

- Train the model using 5-fold cross validation
- Save the model as 'final_cnn_model.keras'

This will generate a model file in the src/mpox/models directory.

This step will take approximately an hour to complete. Alternatively, you can access the data generated by the model through this link: https://drive.google.com/file/d/1fzZ5aBODV9wzLW3VAQFYr-EGx6x-ru-t/view?usp=sharing

If you decide to download the data, remember to unzip the file and relocate the “my_dir” folder and the file named “final_cnn_model.keras” inside the root directory “2024_3_PROJECT_08”.

### 7. Train the second model

```bash
python ./src/stages/3_1_train_pca_rf.py
```

### 8. (Optional) Test the model

To evaluate the model's performance:

Part 1: Skin diseases Classification

```bash
python ./src/mpox/test.py
```

Part 2: Stages Classification

Add image data to the file located at ./src/stages/test_data and run the following command:

```bash
python ./src/stages/3_2_test_pca_rf.py
```

### Requirements

All requirements will be installed requirements.txt

- Python 3.8+
- VSCode
- Required packages:
  - TensorFlow
  - OpenCV
  - Flask
  - NumPy
  - Pandas
  - scikit-learn
  - Jobliv
  - keras-turner
  - kagglehub
  - Pillow
  - keras==3.6.0

<a name="repro"></a>

## 3 Reproduction

1. Go to the root of the project file, then execute the command below:

```bash
python ./src/mpox/app.py

```

2. The application will be available at: http://127.0.0.1:5000
3. Open the application
4. Upload an image file representing any skin disease. (Optional: Sample skin disease can be obtained from './src/mpox/resources/')
5. Results will be displayed on the website's main page.

Note: This application assumes all input is a valid image of skin disease.

<a name="reference"></a>

## 4 Reference

Paul, J., Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0), Kaggle. Available
at: https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20.

<a name="Troubleshooting"></a>

## 5 Troubleshooting

If you encounter module import errors, you'll need to install the missing packages. Here are common errors and their solutions:

- Common error messages and their fixes:

```bash
- `ModuleNotFoundError: No module named 'cv2'` → Run `pip install opencv-python`
- `ModuleNotFoundError: No module named 'PIL'` → Run `pip install Pillow`
- `ModuleNotFoundError: No module named 'scipy'` → Run `pip install scipy`
- `ModuleNotFoundError: No module named 'pandas'` → Run `pip install pandas`
- `ModuleNotFoundError: No module named 'sklearn'` → Run `pip install scikit-learn`
- `ModuleNotFoundError: No module named 'keras_tuner'` → Run `pip install keras-tuner`
- `ModuleNotFoundError: No module named 'flask'` → Run `pip install flask`
- `ModuleNotFoundError: No module named 'keras'` → Run `pip install keras`
```
