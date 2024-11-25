import cv2
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

source_directory = './src/severity/augmented_data'

transformed_features = [] 
classes = [] 

for severity_level in os.listdir(source_directory):
    severity_path = os.path.join(source_directory, severity_level)

    if os.path.isdir(severity_path):
        print(f"Extracting {severity_level}")

        images = os.listdir(severity_path)
        print(f"Found {len(images)} images in {severity_level}")
        
        for img in images:
            img_path = os.path.join(severity_path, img)
            print(f"Processing image: {img}")

            # Ensure the file is an image
            if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Skipping non-image file: {img}")
                continue

            # Read and process the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Resize for consistent processing
            image = cv2.resize(image, (128, 128))
            
            channels = cv2.split(image)
            level_features = []
            for channel in channels:
                fourierTransform = np.fft.fft2(channel)
                fourierShift = np.fft.fftshift(fourierTransform)
                magnitude = 20 * np.log(np.abs(fourierShift) + 1e-10)
                level_features.extend(magnitude.flatten())
            
    
            transformed_features.append(level_features) 
            classes.append(severity_level) 


transformed_features = np.array(transformed_features)
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50)
features_reduction = pca.fit_transform(transformed_features)

output = [
    [severity] + list(features)
    for severity, features in zip(classes, features_reduction)
]

cols = ['Severity'] + [f'Principal_Components_{i}' for i in range(features_reduction.shape[1])]
output_csv = './src/severity/output_pca.csv'
df = pd.DataFrame(severity_level_information, columns=cols)
df.to_csv(output_csv, index=False)
print(f"Features saved to {output_csv}")