import numpy as np
import os
from skimage import io
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

# Function to calculate properties of an image
def calculate_image_properties(image):
    # Convert image to grayscale for calculating brightness and contrast
    grayscale_image = rgb2gray(image)
    
    # Calculate brightness (mean intensity in grayscale)
    brightness = np.mean(grayscale_image)
    
    # Calculate RMS contrast (standard deviation of pixel intensities)
    contrast = np.std(grayscale_image)
    
    # Calculate mean relative intensity for RGB channels
    mean_red = np.mean(image[:, :, 0])
    mean_green = np.mean(image[:, :, 1])
    mean_blue = np.mean(image[:, :, 2])
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'mean_red': mean_red,
        'mean_green': mean_green,
        'mean_blue': mean_blue
    }

# Function to load images from a dataset directory and calculate their properties
def load_and_calculate_properties(dataset_path):
    image_properties = []
    image_paths = []
    
    # Iterate over all image files in the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):  # Assuming image formats are PNG or JPG
                image_path = os.path.join(root, file)
                image = io.imread(image_path)
                
                # Rescale image to [0, 1] range
                image = rescale_intensity(image, in_range=(0, 255), out_range=(0, 1))
                
                # Calculate properties of the image
                properties = calculate_image_properties(image)
                image_properties.append(properties)
                image_paths.append(image_path)
    
    return image_properties, image_paths

# Load the datasets and calculate image properties
train_dataset_path = '/Users/ximenamoure/Desktop/drift_last/reference_dataset'  # Replace with your train dataset path
test_dataset_path = '/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split0'  # Replace with your test dataset path

# Calculate properties for train and test datasets
train_properties, train_image_paths = load_and_calculate_properties(train_dataset_path)
test_properties, test_image_paths = load_and_calculate_properties(test_dataset_path)

# Convert properties to numpy arrays for easier analysis
train_brightness = np.array([prop['brightness'] for prop in train_properties])
train_contrast = np.array([prop['contrast'] for prop in train_properties])
train_mean_red = np.array([prop['mean_red'] for prop in train_properties])

test_brightness = np.array([prop['brightness'] for prop in test_properties])
test_contrast = np.array([prop['contrast'] for prop in test_properties])
test_mean_red = np.array([prop['mean_red'] for prop in test_properties])

# Compare distributions of properties (you can visualize them using histograms)
plt.figure(figsize=(12, 6))

# Brightness
plt.subplot(1, 3, 1)
plt.hist(train_brightness, bins=30, alpha=0.5, label='Train Brightness')
plt.hist(test_brightness, bins=30, alpha=0.5, label='Test Brightness')
plt.title('Brightness Distribution')
plt.legend()

# Contrast
plt.subplot(1, 3, 2)
plt.hist(train_contrast, bins=30, alpha=0.5, label='Train Contrast')
plt.hist(test_contrast, bins=30, alpha=0.5, label='Test Contrast')
plt.title('Contrast Distribution')
plt.legend()

# Mean Red Intensity
plt.subplot(1, 3, 3)
plt.hist(train_mean_red, bins=30, alpha=0.5, label='Train Mean Red Intensity')
plt.hist(test_mean_red, bins=30, alpha=0.5, label='Test Mean Red Intensity')
plt.title('Mean Red Intensity Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Find outliers in the test dataset based on the difference from the mean of train dataset
mean_train_brightness = np.mean(train_brightness)
mean_train_contrast = np.mean(train_contrast)
mean_train_mean_red = np.mean(train_mean_red)

# Calculate z-scores for the test dataset (difference from the mean of train set)
z_scores_brightness = np.abs(test_brightness - mean_train_brightness) / np.std(train_brightness)
z_scores_contrast = np.abs(test_contrast - mean_train_contrast) / np.std(train_contrast)
z_scores_mean_red = np.abs(test_mean_red - mean_train_mean_red) / np.std(train_mean_red)

# Identify top 10 outlier images based on brightness, contrast, and mean red intensity
top_n = 10
outlier_indices_brightness = np.argsort(z_scores_brightness)[-top_n:]
outlier_indices_contrast = np.argsort(z_scores_contrast)[-top_n:]
outlier_indices_mean_red = np.argsort(z_scores_mean_red)[-top_n:]

# Print or save the paths of the outlier images
print("Top 10 Brightness Outliers:")
for idx in outlier_indices_brightness:
    print(test_image_paths[idx])

print("\nTop 10 Contrast Outliers:")
for idx in outlier_indices_contrast:
    print(test_image_paths[idx])

print("\nTop 10 Mean Red Intensity Outliers:")
for idx in outlier_indices_mean_red:
    print(test_image_paths[idx])
