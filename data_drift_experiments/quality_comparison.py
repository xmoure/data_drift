import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load images from a folder path
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])  # Resize if necessary
    return img / 255.0  # Normalize to [0, 1]

# Metric 1: Sharpness using Laplacian variance
# Metric 1: Sharpness using Laplacian variance
def calculate_sharpness(image):
    # Convert the normalized image back to the uint8 format required by OpenCV
    img = (image.numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# Metric 2: Brightness using average pixel intensity
def calculate_brightness(image):
    return np.mean(image.numpy())

# Metric 3: Contrast using standard deviation of pixel intensities
def calculate_contrast(image):
    return np.std(image.numpy())

# Metric 4: Color distribution across R, G, and B channels
def calculate_color_distribution(image):
    img = image.numpy()
    red_channel = np.mean(img[..., 0])
    green_channel = np.mean(img[..., 1])
    blue_channel = np.mean(img[..., 2])
    return red_channel, green_channel, blue_channel

# Function to gather all image paths from a split
def gather_image_paths(split_dir):
    image_paths = []
    for root, dirs, files in os.walk(split_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Analyze all images in each split and calculate metrics
def analyze_split(image_paths):
    sharpness_values = []
    brightness_values = []
    contrast_values = []
    red_values = []
    green_values = []
    blue_values = []

    for path in image_paths:
        img = load_image(path)
        sharpness_values.append(calculate_sharpness(img))
        brightness_values.append(calculate_brightness(img))
        contrast_values.append(calculate_contrast(img))
        red, green, blue = calculate_color_distribution(img)
        red_values.append(red)
        green_values.append(green)
        blue_values.append(blue)

    return {
        'sharpness': sharpness_values,
        'brightness': brightness_values,
        'contrast': contrast_values,
        'red': red_values,
        'green': green_values,
        'blue': blue_values
    }

# Function to plot and compare distributions of a metric between splits
def plot_metric_distribution(metrics_dict, metric_name, split_names):
    plt.figure(figsize=(10, 6))
    for split_name in split_names:
        plt.hist(metrics_dict[split_name][metric_name], bins=50, alpha=0.7, label=split_name)
    plt.title(f'{metric_name.capitalize()} Distribution Across Splits')
    plt.xlabel(metric_name.capitalize())
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# For color distribution, plot each channel separately
def plot_color_distribution(metrics_dict, color, split_names):
    plt.figure(figsize=(10, 6))
    for split_name in split_names:
        plt.hist(metrics_dict[split_name][color], bins=50, alpha=0.7, label=split_name)
    plt.title(f'{color.capitalize()} Channel Distribution Across Splits')
    plt.xlabel(f'{color.capitalize()} Channel Intensity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Summarize the metrics for all splits
def summarize_metrics(metrics_dict, split_names):
    for split_name in split_names:
        print(f"\nSummary for {split_name}:")
        for metric, values in metrics_dict[split_name].items():
            print(f"{metric.capitalize()}: Mean = {np.mean(values):.2f}, Std Dev = {np.std(values):.2f}")

# Main code: Define paths to each split directory
split5_dir = '/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split5'  # Adjust the path to your actual split0 directory
split0_dir = '/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split0'  # Add more splits if needed
split1_dir = '/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split1'  # Add more splits if needed

# Collect paths from each split
splits = {
    'split5': gather_image_paths(split5_dir),
    'split0': gather_image_paths(split0_dir),
    'split1': gather_image_paths(split1_dir),
}

# Analyze each split
split_metrics = {}
for split_name, image_paths in splits.items():
    split_metrics[split_name] = analyze_split(image_paths)

# List of splits for comparison
split_names = list(splits.keys())

# Plot comparisons for different metrics
plot_metric_distribution(split_metrics, 'sharpness', split_names)
plot_metric_distribution(split_metrics, 'brightness', split_names)
plot_metric_distribution(split_metrics, 'contrast', split_names)

# Plot color distribution comparisons
plot_color_distribution(split_metrics, 'red', split_names)
plot_color_distribution(split_metrics, 'green', split_names)
plot_color_distribution(split_metrics, 'blue', split_names)

# Summarize metrics for each split
summarize_metrics(split_metrics, split_names)

