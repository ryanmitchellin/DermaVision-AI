from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def augment_data(source, output, num):

    class_dirs = os.listdir(source)
    for class_dir in class_dirs:
        os.makedirs(os.path.join(output, class_dir), exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest'
    )
   
    for class_dir in class_dirs:
        class_generator = datagen.flow_from_directory(
            directory=source,
            target_size=(224, 224),
            batch_size=1,
            classes=[class_dir], 
            class_mode='categorical',
            save_to_dir=os.path.join(output, class_dir),
            save_prefix=f'{class_dir}_',
            save_format='jpeg'
        )
        
        for i in range(num):
            next(class_generator)

if __name__ == "__main__":
    source = "./src/stages/classifiedData"  
    output = "./src/stages/augmented_data"  
    num = 150 

    augment_data(source, output, num)