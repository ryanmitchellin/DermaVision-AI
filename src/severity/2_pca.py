import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import joblib

source = './src/severity/augmented_data'

data = []
labels = []
format = ('.jpg', '.jpeg')

for severity_level in os.listdir(source):
    severity_path = os.path.join(source, severity_level)
    
    if os.path.isdir(severity_path):
        print(f"Reading {severity_level}...")
        
        for img_name in os.listdir(severity_path):
            img_path = os.path.join(severity_path, img_name)
            
            if img_name.lower().endswith(format):
                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.resize(image, (128,128))
                data.append(image.flatten())
                labels.append(severity_level)
            else:
                continue

output_path = './src/severity/pca_output.csv'
data = np.array(data)
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50) # Change here for the number of feature extracted 
features_reduction = pca.fit_transform(data)

severity_level_information = [
    [labels] + list(features) for labels, features in zip(labels, features_reduction)
]

cols = ['Severity'] + [f'Principal_Components_{i}' for i in range(features_reduction.shape[1])]
df = pd.DataFrame(severity_level_information, columns=cols)
df.to_csv(output_path, index=False)
print(f"Features saved to {output_path}")

# Save the PCA Model
joblib.dump(pca, './src/severity/models/2_pca_model.pkl')
print("Final model saved as 2_pca_model.pkl")