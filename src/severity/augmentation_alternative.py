import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import gc

def build_autoencoder(input_shape=(224, 224, 3)):  # Changed from 96x96 to 224x224
    # Increased network capacity for better color handling
    input_img = Input(shape=input_shape)
    
    # Encoder - increased filters
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder - symmetric to encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # Final layer with 3 channels for RGB
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    encoder = Model(input_img, encoded)
    
    # Decoder model
    decoder_input = Input(shape=encoded.shape[1:])
    decoder_layers = autoencoder.layers[-7:]
    decoder_output = decoder_input
    for layer in decoder_layers:
        decoder_output = layer(decoder_output)
    decoder = Model(decoder_input, decoder_output)

    return autoencoder, encoder, decoder

def prepare_small_dataset(data_flow, num_samples=53):
    images = []
    labels = []
    
    for _ in range(num_samples):
        img, label = next(data_flow)
        images.append(img[0])  # Take first image from batch
        labels.append(label[0])
        
        if len(images) >= num_samples:
            break

    return np.array(images), np.array(labels)

def generate_new_images(models, images, original_filenames, num_variations=3):
    autoencoder, encoder, decoder = models
    encoded_imgs = encoder.predict(images)
    new_images = []
    augmented_filenames = []

    for i, original_filename in enumerate(original_filenames):
        encoded_img = encoded_imgs[i:i+1]
        for j in range(num_variations):
            # Reduced noise to preserve colors better
            noise = np.random.normal(0, 0.05, encoded_img.shape)
            noisy_encoded = encoded_img + noise
            new_img = decoder.predict(noisy_encoded)[0]
            # Ensure proper color range
            new_img = np.clip(new_img, 0, 1)
            new_images.append(new_img)
            augmented_filename = f"{os.path.splitext(original_filename)[0]}_aug_{j}.png"
            augmented_filenames.append(augmented_filename)

    return np.array(new_images), augmented_filenames

def save_augmented_images(augmented_images, augmented_filenames, base_path):
    os.makedirs(base_path, exist_ok=True)
    
    for img, filename in zip(augmented_images, augmented_filenames):
        # Convert to uint8 format
        img_uint8 = (img * 255).astype(np.uint8)
        # Save using PIL for better color preservation
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(os.path.join(base_path, filename))

def augment_small_dataset(data_flow):
    try:
        print("Loading and preparing dataset...")
        input_images, labels = prepare_small_dataset(data_flow)
        
        print("Building and training autoencoder...")
        autoencoder, encoder, decoder = build_autoencoder(input_shape=input_images.shape[1:])
        
        autoencoder.fit(
            input_images,
            input_images,
            epochs=30,  # Increased epochs for better color learning
            batch_size=4,
            shuffle=True,
            validation_split=0.2,
            verbose=1
        )
        
        print("Generating new images...")
        original_filenames = [os.path.basename(filepath) for filepath in data_flow.filepaths[:len(input_images)]]
        augmented_images, augmented_filenames = generate_new_images(
            (autoencoder, encoder, decoder),
            input_images,
            original_filenames
        )
        
        del autoencoder, encoder, decoder, labels
        gc.collect()
        
        return augmented_images, augmented_filenames
    
    except Exception as e:
        print(f"Error in augment_small_dataset: {str(e)}")
        raise

def augment_each_class_folder(base_dir='./src/severity/kaggleClassified', 
                            output_dir='./src/severity/augmented_data'):
    print("Starting color image augmentation process")
    
    # Modified data generator settings
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,  # Reduced rotation for better color preservation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    class_folders = os.listdir(base_dir)
    for class_name in class_folders:
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            print(f"\nProcessing class: {class_name}")
            
            data_flow = data_gen.flow_from_directory(
                base_dir,
                target_size=(224, 224),  # Changed from 96x96 to 224x224
                batch_size=1,
                color_mode='rgb',
                classes=[class_name],
                class_mode='categorical',
                shuffle=True
            )

            augmented_images, augmented_filenames = augment_small_dataset(data_flow)
            class_output_dir = os.path.join(output_dir, class_name)
            save_augmented_images(augmented_images, augmented_filenames, base_path=class_output_dir)
            print(f"Finished processing {class_name}")

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        augment_each_class_folder()
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        print(f"\nTotal processing time: {int(minutes)} minutes and {int(seconds)} seconds")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")