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

def create_autoencoder(img_shape=(224, 224, 3)):
    input = Input(shape=img_shape)

    encode = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    encode = MaxPooling2D((2, 2), padding='same')(encode)
    encode = Conv2D(32, (3, 3), activation='relu', padding='same')(encode)
    encode = MaxPooling2D((2, 2), padding='same')(encode)
    encode = Conv2D(16, (3, 3), activation='relu', padding='same')(encode)
    bottleneck = MaxPooling2D((2, 2), padding='same')(encode)  

    decode = Conv2D(16, (3, 3), activation='relu', padding='same')(bottleneck)
    decode = UpSampling2D((2, 2))(decode)
    decode = Conv2D(32, (3, 3), activation='relu', padding='same')(decode)
    decode = UpSampling2D((2, 2))(decode)
    decode = Conv2D(64, (3, 3), activation='relu', padding='same')(decode)
    decode = UpSampling2D((2, 2))(decode)
    
    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decode)

    autoencoder_model = Model(input, output)
    autoencoder_model.compile(optimizer='adam', loss='mse')

    encoder_model = Model(input, bottleneck)

    decoder_input = Input(shape=bottleneck.shape[1:])
    layers = autoencoder_model.layers[-7:]
    decode_output = decoder_input
    for decode_layer in layers:
        decode_output = decode_layer(decode_output)
    decoder_model = Model(decoder_input, decode_output)

    return autoencoder_model, encoder_model, decoder_model

def create_dataset(data_flow, sample_size=53):
    img_samples = []
    label_samples = []
    
    for samples in range(sample_size):
        img, label = next(data_flow)
        img_samples.append(img[0]) 
        label_samples.append(label[0])
        if len(img_samples) >= sample_size:
            break

    return np.array(img_samples), np.array(label_samples)

def create_samples(models, images, source_filenames, num_variations=3):
    autoencoder, encoder, decoder = models
    img_result = encoder.predict(images)
    output_image = []
    output_files = []

    for i, filename in enumerate(source_filenames):
        list = img_result[i:i+1]
        for j in range(num_variations):
            noise = np.random.normal(0, 0.05, list.shape)
            noisy_encoded = list + noise
            output = decoder.predict(noisy_encoded)[0]
            output = np.clip(output, 0, 1)
            output_image.append(output)
            file = f"{os.path.splitext(filename)[0]}_aug_{j}.png"
            output_files.append(file)

    return np.array(output_image), output_files

def save_images(augmented_images, augmented_filenames, base_path):
    os.makedirs(base_path, exist_ok=True)
    
    for img, filename in zip(augmented_images, augmented_filenames):
        img_uint8 = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(os.path.join(base_path, filename))

def augment_small_dataset(data_flow):
    try:
        input_images, labels = create_dataset(data_flow)

        autoencoder, encoder, decoder = create_autoencoder()
        
        autoencoder.fit(
            input_images,
            input_images,
            epochs=30,  
            batch_size=4,
            shuffle=True,
            validation_split=0.2,
            verbose=1
        )
        
        source_filenames = [os.path.basename(filepath) for filepath in data_flow.filepaths[:len(input_images)]]
        augmented_images, augmented_filenames = create_samples(
            (autoencoder, encoder, decoder),
            input_images,
            source_filenames
        )
        
        del autoencoder, encoder, decoder, labels
        gc.collect()
        
        return augmented_images, augmented_filenames
    
    except Exception as e:
        print(f"Error in augment_small_dataset: {str(e)}")
        raise

def augment_files(sourcepath, outputpath):
    
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,  
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    class_folders = os.listdir(sourcepath)
    for class_name in class_folders:
        class_path = os.path.join(sourcepath, class_name)
        if os.path.isdir(class_path):
            print(f"\nProcessing class: {class_name}")
            
            data_flow = data_gen.flow_from_directory(
                sourcepath,
                target_size=(224, 224), 
                batch_size=1,
                color_mode='rgb',
                classes=[class_name],
                class_mode='categorical',
                shuffle=True
            )

            images, files = augment_small_dataset(data_flow)
            files_output = os.path.join(outputpath, class_name)
            save_images(images, files, base_path=files_output)
            print(f"{class_name} Completeed")

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        sourcepath="./src/severity/kaggleClassified"
        outputpath="./src/severity/augmented_autoencoder"
        augment_files(sourcepath, outputpath)
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        print(f"\nTotal processing time: {int(minutes)} minutes and {int(seconds)} seconds")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")