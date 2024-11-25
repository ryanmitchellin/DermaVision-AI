# DermaVision AI
This project is designed as a tool to assist users in identifying skin diseases by analyzing uploaded images. Upon submitting a picture, the system provides feedback on the diagnosed condition and, in cases where the disease is classified as mpox, also evaluate its severity level. 


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/Ea_oAN9CB_hBi3OZi4DQiaIBHtT-s4eWAng1HMl6Hh85kA?e=C4xfla) | [Slack channel](https://app.slack.com/client/T07K7SWL5A4/C07JS14AD43) | [Project report](https://www.overleaf.com/project/66d0b0532317a8cadc2e64f1) |

## Video/demo/GIF

Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.

## Table of Contents

1. [Demo](#demo)
2. [Installation](#installation)
3. [Reproducing this project](#repro)
4. [Guidance](#guide)

<a name="demo"></a>

## 1. Example demo

Main Features:

- Image upload and preview functionality
- Diagnosis results of skin conditions
- Information about the skin condition
- Educational resources about the skin condition

### What to find where

Explain briefly what files are found where

# TO DO: Modify this 
```bash
repository
├── static/                        # Frontend assets
│ ├── style.css                    # CSS for styling
│ ├── script.js                    # JavaScript for image handling
│ └── about.js                     # JavaScript for the about page
├── templates/                     # HTML templates
│ ├── index.html                   # Main diagnosis page
│ └── about.html                   # About Page for Disease information and team details
├── src/                           # Backend source code
│ └── mpox/                        # Part 1: Skin Disease Classification
│   │   ├── app.py                 # Flask server and routing
│   │   ├── train.py               # Training the model
│   │   ├── test.py                # Testing the model
│   │   ├── output.py              # Prediction output
│ └── severity/                    # Part 2: Skin Disease Classification
│   │   ├── augmented_data         # Data output from keras_augmentation.py 
│   │   │  ├── ...                 # Files storing images, each categorized into their class
│   │   ├── kaggleClassified       # Files containing classified model data
│   │   │  ├── ...                 # Images categorized into their class
│   │   ├── keras_augmentation.py  # Augmenting Limited Data
│   │   ├── train_simple.py        # Training the model
│   │   ├── test.py                # Testing the model
│ └── resources/                   # Resource files and test data
└── README.md                      # Project documentation
```

<a name="installation"></a>

## 2a. Installation: Part 1
### 1. Clone the repository

```bash
git clone git@github.com:sfu-cmpt340/2024_3_project_08.git
cd 2024_3_project_08
```

### 2. Set Up environment

```bash
conda env create -f requirements.yml
conda activate amazing
```

### 3. Download the dataset

- Go to the source  https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset/data (Nafin, n.d.).
- Download the dataset by executing the provided code in a standalone .py file (Note: Make sure you don't download the dataset as zip as it messes up the pathway of the inner files).
- The folder's path will be displayed in the terminal. Relocate it to ./src/mpox/.
- Ensure that the file path './src/mpox/mpox-skin-lesion-dataset-version-20-msld-v20/versions/4' exists. 

### 4. Train the model

```bash
cd src/mpox
python src/mpox/train.py
```

#### key features:

- Train the model using 5-fold cross validation
- Save the model as 'final_cnn_model.keras'

This will generate a model file in the src/mpox/models directory.

### 5. Run the application

```bash
cd src/mpox
python app.py
```

The application will be available at: http://127.0.0.1:5000

### 6. (Optional) Test the model

To evaluate the model's performance:

### Requirements

- Python 3.8+
- Conda package manager
- Required packages (installed via requirements.yml):
  - TensorFlow
  - OpenCV
  - Flask
  - NumPy
  - scikit-learn

<a name="repro"></a>

## 3. Reproduction

Demonstrate how your work can be reproduced, e.g. the results in your report.

```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```

Data can be found at ...
Output will be saved in ...

<a name="guide"></a>

## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
  - Do NOT use history re-editing (rebase)
  - Commit messages should be informative:
    - No: 'this should fix it', 'bump' commit messages
    - Yes: 'Resolve invalid API call in updating X'
  - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
  - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/)

## 5. Reference

Nafin, M., Monkeypox Skin Lesion Dataset, Kaggle. Available at:
https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset/data.

