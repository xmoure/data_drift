import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os
from glob import glob
import random
import pickle
from tqdm import tqdm
import tensorflow_addons as tfa

# Load the encoder model
encoder = load_model('/Users/ximenamoure/Desktop/drift_last/split5_training_autoencoder_v3/encoder_model.h5')

# Gaussian blur function (adjust if necessary)
def gaussian_blur(image, kernel_size=5, sigma=1.0):
    def gaussian_kernel(size: int, std: float):
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        x = tf.exp(-0.5 * tf.square(x / std))
        gaussian_kernel_1d = x / tf.reduce_sum(x)
        gaussian_kernel_2d = tf.tensordot(gaussian_kernel_1d, gaussian_kernel_1d, axes=0)
        gaussian_kernel_2d = gaussian_kernel_2d / tf.reduce_sum(gaussian_kernel_2d)
        return gaussian_kernel_2d

    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    num_channels = tf.shape(image)[-1]
    kernel = tf.repeat(kernel, repeats=num_channels, axis=-1)
    image = image[tf.newaxis, ...]
    blurred_image = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(blurred_image, axis=0)



def rotate_image(image, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    radians = angle * (3.14159 / 180)  # Convert degrees to radians
    return tfa.image.rotate(image, radians)


def random_cutout(image, max_cutout_size=30):
    # Get the image dimensions
    image_shape = tf.shape(image)
    img_height = image_shape[0]
    img_width = image_shape[1]

    # Define the cutout dimensions
    cutout_height = tf.random.uniform([], minval=10, maxval=max_cutout_size, dtype=tf.int32)
    cutout_width = tf.random.uniform([], minval=10, maxval=max_cutout_size, dtype=tf.int32)
    
    # Randomly choose the top left corner for the cutout area
    cutout_x = tf.random.uniform([], minval=0, maxval=img_width - cutout_width, dtype=tf.int32)
    cutout_y = tf.random.uniform([], minval=0, maxval=img_height - cutout_height, dtype=tf.int32)
    
    # Create a mask that is 1 everywhere except the cutout area, which is 0
    cutout_shape = [cutout_height, cutout_width, 3]
    padding_dims = [
        [cutout_y, img_height - cutout_y - cutout_height],
        [cutout_x, img_width - cutout_x - cutout_width],
        [0, 0]
    ]
    cutout_mask = tf.pad(tf.zeros(cutout_shape, dtype=tf.float32), padding_dims, constant_values=1)
    
    # Apply the mask to the image
    image = image * cutout_mask
    return image



def salt_and_pepper_noise(image, probability=0.05):
    noise = tf.random.uniform(tf.shape(image), minval=0, maxval=1)
    salt = tf.cast(noise < probability / 2, tf.float32)
    pepper = tf.cast(noise > 1 - probability / 2, tf.float32)
    return image * (1 - salt) + pepper


def convert_to_grayscale(image):
    grayscale_image = tf.image.rgb_to_grayscale(image)
    return tf.image.grayscale_to_rgb(grayscale_image)



""" # Function to apply drift to an image
def apply_drift_image(image, drift_percentage=0.5):
    if random.random() < drift_percentage:
        image = tf.image.adjust_brightness(image, delta=random.uniform(-1.0, 1.0))
        image = tf.image.adjust_contrast(image, contrast_factor=random.uniform(0.2, 2.0))
        image = gaussian_blur(image, kernel_size=7, sigma=random.uniform(0.2, 2.0))
        image_shape = tf.shape(image)
        crop_height = tf.cast(0.9 * tf.cast(image_shape[0], tf.float32), tf.int32)
        crop_width = tf.cast(0.9 * tf.cast(image_shape[1], tf.float32), tf.int32)
        if image_shape[0] >= crop_height and image_shape[1] >= crop_width:
            image = tf.image.random_crop(image, size=[crop_height, crop_width, 3])
            image = tf.image.resize(image, [128, 128])
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.15, dtype=tf.float32)
        image = tf.add(image, noise)
    return image """

# Function to apply more intense drift to an image
def apply_drift_image(image, drift_percentage=0.5):
    if random.random() <= drift_percentage:
        # Apply strong random brightness adjustment
        image = tf.image.adjust_brightness(image, delta=random.uniform(-2.5, 2.5))  # Increase brightness delta range
        
        # Apply strong random contrast adjustment
        image = tf.image.adjust_contrast(image, contrast_factor=random.uniform(0.05, 4.0))  # Increase contrast factor range

        # Apply color jitter: adjust hue and saturation
        image = tf.image.adjust_hue(image, delta=random.uniform(-0.4, 0.4))  # Randomly shift hue by up to ±0.2
        image = tf.image.adjust_saturation(image, saturation_factor=random.uniform(0.3, 3.5))  # Randomly adjust saturation

        # Randomly apply additional transformations
        if random.random() < 0.5:
            image = gaussian_blur(image, kernel_size=25, sigma=random.uniform(2.0, 5.0))
        if random.random() < 0.4:
            image = rotate_image(image, max_angle=45)
        if random.random() < 0.5:
            image = random_cutout(image, max_cutout_size=40)
        if random.random() < 0.5:
            image = salt_and_pepper_noise(image, probability=0.05)
        if random.random() < 0.3:
            image = convert_to_grayscale(image)
        
        # Apply Gaussian blur with a larger kernel and stronger sigma
        #image = gaussian_blur(image, kernel_size=15, sigma=random.uniform(2.0, 5.0))  # Larger blur kernel and sigma
        
        # Random crop with a higher percentage removed
        image_shape = tf.shape(image)
        crop_height = tf.cast(0.7 * tf.cast(image_shape[0], tf.float32), tf.int32)  # Crop to 80% of original height
        crop_width = tf.cast(0.7 * tf.cast(image_shape[1], tf.float32), tf.int32)   # Crop to 80% of original width
        if image_shape[0] >= crop_height and image_shape[1] >= crop_width:
            image = tf.image.random_crop(image, size=[crop_height, crop_width, 3])
            image = tf.image.resize(image, [128, 128])  # Resize back to original size
        
        # Add more random noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.45, dtype=tf.float32)  # Increase noise stddev
        image = tf.add(image, noise)

        image = tf.image.rot90(image, k=random.randint(1, 3))  # Random rotation by 90, 180, or 270 degrees
        
        # Clip the pixel values to keep them in [0, 1] range after transformations
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image


# Process images with drift applied in batches
def process_images_in_batches(image_paths, drift_percentage=0.5, batch_size=32, apply_transformation = False):
    num_images = len(image_paths)
    embeddings = []

    for i in tqdm(range(0, num_images, batch_size), desc="Processing images in batches"):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess the batch of images
        images = [tf.image.decode_png(tf.io.read_file(path), channels=3) for path in batch_paths]
        images = [tf.image.resize(image, [128, 128]) / 255.0 for image in images]  # Normalize to [0, 1]
        if(apply_transformation):
            images = [apply_drift_image(image, drift_percentage) for image in images]  # Apply drift to each image

        # Stack the images into a single batch tensor
        batch_images = tf.stack(images)

        # Pass the batch through the encoder model to get embeddings
        batch_embeddings = encoder.predict(batch_images)
        
        # Append the embeddings
        embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings into a single NumPy array
    return np.vstack(embeddings)

def load_and_downsample_image_paths(data_folder):
    occupied = glob(os.path.join(data_folder, "[!empty]*/*.png"))
    empty = glob(os.path.join(data_folder, "empty/*.png"))

    print(f"Occupied samples: {len(occupied)}")
    print(f"Empty samples: {len(empty)}")

    # Downsample the empty class to match the number of occupied samples
    if len(empty) > len(occupied):
        rng = np.random.default_rng(seed=42)
        rng.shuffle(empty)
        empty = empty[:len(occupied)]

    print(f"Downsampled empty samples: {len(empty)}")

    # Combine paths
    paths = np.array(occupied + empty)

    return paths

# Load image paths
image_folder = '/Users/ximenamoure/Desktop/drift_last/reference_dataset'
image_paths = load_and_downsample_image_paths(image_folder)
apply_trans = True

# Process images and get embeddings with drift applied to 50% of the images
embeddings_with_drift = process_images_in_batches(image_paths, drift_percentage=0.50, batch_size=32, apply_transformation= apply_trans)

# Now you have embeddings for each image, with drift applied to the specified percentage
print("Embeddings shape:", embeddings_with_drift.shape)

print("Saving embeddings")
with open('/Users/ximenamoure/Desktop/drift_last/embeddings_autoencoder_v3/ref_embeddings_autov3_40_trans.pkl', 'wb') as f:
    pickle.dump(embeddings_with_drift, f)
