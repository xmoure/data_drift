import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage import io, exposure, filters
import numpy as np

# Number of bins for the histogram
NUM_BINS = 256

# Function to compute brightness (mean pixel value)
def compute_brightness(image_path):
    image = io.imread(image_path)
    return image.mean()

# Function to compute contrast (standard deviation of pixel values)
def compute_contrast(image_path):
    image = io.imread(image_path)
    return image.std()

# Function to compute sharpness using a Laplacian filter
def compute_sharpness(image_path):
    image = io.imread(image_path, as_gray=True)
    return filters.laplace(image).var()

# Function to compute color histogram with fixed number of bins
def compute_color_histogram(image_path):
    image = io.imread(image_path)
    if image.ndim == 2:  # Grayscale image
        hist, _ = np.histogram(image, bins=NUM_BINS, range=(0, 256))
        return hist
    elif image.ndim == 3:  # RGB image
        hist_r, _ = np.histogram(image[..., 0], bins=NUM_BINS, range=(0, 256))
        hist_g, _ = np.histogram(image[..., 1], bins=NUM_BINS, range=(0, 256))
        hist_b, _ = np.histogram(image[..., 2], bins=NUM_BINS, range=(0, 256))
        return hist_r + hist_g + hist_b

# Paths to your datasets
reference_dataset = '/Users/ximenamoure/Desktop/drift_last/reference_dataset'
new_dataset = '/Users/ximenamoure/Desktop/split6'


# Load image paths recursively from subfolders
reference_image_paths = glob(os.path.join(reference_dataset, '**/*.png'), recursive=True)
new_image_paths = glob(os.path.join(new_dataset, '**/*.png'), recursive=True)


# Downsample reference dataset if necessary
if len(reference_image_paths) > len(new_image_paths):
    np.random.seed(42)
    sampled_paths = np.random.choice(reference_image_paths, size=len(new_image_paths), replace=False)
else:
    sampled_paths = reference_image_paths


# Initialize DataFrames to store properties
reference_stats = pd.DataFrame(sampled_paths, columns=['filename'])
new_stats = pd.DataFrame(new_image_paths, columns=['filename'])

# Compute properties for reference dataset
reference_stats['brightness'] = reference_stats['filename'].apply(compute_brightness)
reference_stats['contrast'] = reference_stats['filename'].apply(compute_contrast)
reference_stats['sharpness'] = reference_stats['filename'].apply(compute_sharpness)
reference_stats['color_histogram'] = reference_stats['filename'].apply(compute_color_histogram)

# Compute properties for new dataset
new_stats['brightness'] = new_stats['filename'].apply(compute_brightness)
new_stats['contrast'] = new_stats['filename'].apply(compute_contrast)
new_stats['sharpness'] = new_stats['filename'].apply(compute_sharpness)
new_stats['color_histogram'] = new_stats['filename'].apply(compute_color_histogram)

# Plotting helper function
def plot_histogram(property_name, reference_stats, new_stats, bins=50):
    plt.figure(figsize=(12, 6))
    plt.hist(reference_stats[property_name], bins=bins, alpha=0.5, label='Reference')
    plt.hist(new_stats[property_name], bins=bins, alpha=0.5, label='New')
    plt.xlabel(property_name.capitalize())
    plt.ylabel('Frequency')
    plt.title(f'{property_name.capitalize()} Distribution')
    plt.legend(loc='upper right')
    plt.show()

# Plot distributions for each property
properties = ['brightness', 'contrast', 'sharpness']
for prop in properties:
    plot_histogram(prop, reference_stats, new_stats)

# Compare color histograms (use L2 norm to compare histograms)
def compare_color_histograms(reference_stats, new_stats):
    ref_histograms = np.vstack(reference_stats['color_histogram'])
    new_histograms = np.vstack(new_stats['color_histogram'])
    ref_hist_mean = np.mean(ref_histograms, axis=0)
    new_hist_mean = np.mean(new_histograms, axis=0)
    distance = np.linalg.norm(ref_hist_mean - new_hist_mean)
    print(f'Color histogram distance: {distance}')

compare_color_histograms(reference_stats, new_stats)


# Function to plot color histograms
def plot_color_histograms(reference_hist, new_hist, title):
    plt.figure(figsize=(12, 6))
    plt.plot(reference_hist, color='blue', label='Reference')
    plt.plot(new_hist, color='orange', label='New')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()


# Compute mean color histograms
ref_hist_mean = np.mean(np.vstack(reference_stats['color_histogram']), axis=0)
new_hist_mean = np.mean(np.vstack(new_stats['color_histogram']), axis=0)

# Plot overall color histograms
plot_color_histograms(ref_hist_mean, new_hist_mean, 'Overall Color Histogram')

# Compute and plot individual color channel histograms
def compute_channel_histograms(image_path, num_bins=256):
    image = io.imread(image_path)
    hist_r, _ = np.histogram(image[..., 0], bins=num_bins, range=(0, 256))
    hist_g, _ = np.histogram(image[..., 1], bins=num_bins, range=(0, 256))
    hist_b, _ = np.histogram(image[..., 2], bins=num_bins, range=(0, 256))
    return hist_r, hist_g, hist_b

reference_channel_histograms = [compute_channel_histograms(img) for img in reference_image_paths]
new_channel_histograms = [compute_channel_histograms(img) for img in new_image_paths]

ref_hist_r_mean = np.mean([hist[0] for hist in reference_channel_histograms], axis=0)
ref_hist_g_mean = np.mean([hist[1] for hist in reference_channel_histograms], axis=0)
ref_hist_b_mean = np.mean([hist[2] for hist in reference_channel_histograms], axis=0)

new_hist_r_mean = np.mean([hist[0] for hist in new_channel_histograms], axis=0)
new_hist_g_mean = np.mean([hist[1] for hist in new_channel_histograms], axis=0)
new_hist_b_mean = np.mean([hist[2] for hist in new_channel_histograms], axis=0)

# Plot individual color channel histograms
plot_color_histograms(ref_hist_r_mean, new_hist_r_mean, 'Red Channel Histogram')
plot_color_histograms(ref_hist_g_mean, new_hist_g_mean, 'Green Channel Histogram')
plot_color_histograms(ref_hist_b_mean, new_hist_b_mean, 'Blue Channel Histogram')

