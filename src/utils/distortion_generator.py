from keras.preprocessing.image import ImageDataGenerator
from src.utils.generators import SiameseGeneratorDistortions
import os


# DISTORTION_GENERATORS = [
#     ImageDataGenerator(rescale=1.0 / 255, brightness_range=[0.2, 0.66]),
#     ImageDataGenerator(rescale=1.0 / 255, brightness_range=[1.5, 5.0]),
#     ImageDataGenerator(rescale=1.0 / 255, rotation_range=90.0),
#     ImageDataGenerator(rescale=1.0 / 255, shear_range=90.0),
#     ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.5),
#     ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=SiameseGeneratorDistortions.apply_blur),
#     ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=SiameseGeneratorDistortions.apply_blob_overlay),
#     ImageDataGenerator(),
# ]

DISTORTION_GENERATORS = [
    ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=SiameseGeneratorDistortions.apply_blur),
]

ORIGINAL_IMAGES_DIR = 'datasets/distortions/originals'
DISTORTED_IMAGES_PARENT_DIR = 'datasets/distortions/variants_v2'
# DISTORTION_NAMES = ['underexposure', 'overexposure', 'rotation', 'shear', 'zoom', 'blur', 'blob_overlay', 'original']
DISTORTION_NAMES = ['blur_v2']

if __name__ == "__main__":

    distorted_images_dirs = [os.path.join(DISTORTED_IMAGES_PARENT_DIR, name) for name in DISTORTION_NAMES]
    for dir_path in distorted_images_dirs:
        os.makedirs(dir_path, exist_ok=True)


    for i, generator in enumerate(DISTORTION_GENERATORS):
        distortion_name = DISTORTION_NAMES[i]
        distorted_images_dir = distorted_images_dirs[i]
        datagen = generator.flow_from_directory(
            directory=ORIGINAL_IMAGES_DIR,
            target_size=(480, 640),  # Specify target size
            batch_size=1,
            class_mode=None,  # No labels, only images
            save_to_dir=distorted_images_dir,  # Save augmented images to the corresponding folder
            save_prefix='',
            save_format='jpg'
        )
        
        # Generate and save distorted images
        num_batches = len(datagen)
        for _ in range(num_batches):
            batch = next(datagen)
