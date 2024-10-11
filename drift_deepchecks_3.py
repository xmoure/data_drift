from deepchecks.vision.checks import ImagePropertyDrift
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper
from deepchecks.core.check_result import CheckResult
from deepchecks.vision import classification_dataset_from_directory
import numpy as np
import os
import pandas as pd
import shutil

# Path to train (reference) and test (new) datasets
dataset_path1 = '/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split5'
dataset_path2 = '/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split0'

# Load the datasets
train_ds = classification_dataset_from_directory(
    root=dataset_path1,
    object_type='VisionData',
    image_extension='png'
)

test_ds = classification_dataset_from_directory(
    root=dataset_path2,
    object_type='VisionData',
    image_extension='png'
)

# Rename datasets for report
train_ds.name = "Train Split"
test_ds.name = "Test Split"

# Image paths for reference later
train_image_paths = train_ds.batch_loader.dataset.images
test_image_paths = test_ds.batch_loader.dataset.images

# Initialize ImagePropertyDrift check with default image properties
image_property_drift = ImagePropertyDrift(n_samples=50000)

# Run the drift check between the train and test datasets
check_result = image_property_drift.run(train_ds, test_ds)

# Save the drift check results
check_result.save_as_html('image_property_drift_results.html')

# Access the result value for each property drift score
drift_results = check_result.value

# Print drift results for each property
for property_name, result in drift_results.items():
    print(f"Property: {property_name}, Drift Score: {result['Drift score']}, Method: {result['Method']}")

# Initialize containers for storing drifted images by property
drifted_images = {property_name: [] for property_name in drift_results.keys()}

# Assuming you've already extracted the properties from the datasets (train and test)
train_properties = image_property_drift._train_properties
test_properties = image_property_drift._test_properties

# Directory to save drifted images by property
base_outliers_dir = 'drifted_images_by_property'
os.makedirs(base_outliers_dir, exist_ok=True)

# Loop through properties and set thresholds for drift detection
for property_name, result in drift_results.items():
    # Handle drift for each property
    if property_name == 'Area':
        drift_threshold = 10
        for idx, area in enumerate(test_properties['Area']):
            if abs(area - np.mean(train_properties['Area'])) > drift_threshold:
                drifted_images['Area'].append(test_image_paths[idx])

    elif property_name == 'Brightness':
        drift_threshold = np.std(train_properties['Brightness']) * 2
        for idx, brightness in enumerate(test_properties['Brightness']):
            if abs(brightness - np.mean(train_properties['Brightness'])) > drift_threshold:
                drifted_images['Brightness'].append(test_image_paths[idx])

    elif property_name == 'RMS Contrast':
        drift_threshold = np.std(train_properties['RMS Contrast']) * 2
        for idx, contrast in enumerate(test_properties['RMS Contrast']):
            if abs(contrast - np.mean(train_properties['RMS Contrast'])) > drift_threshold:
                drifted_images['RMS Contrast'].append(test_image_paths[idx])

    elif property_name == 'Mean Red Relative Intensity':
        drift_threshold = np.std(train_properties['Mean Red Relative Intensity']) * 2
        for idx, red_intensity in enumerate(test_properties['Mean Red Relative Intensity']):
            if abs(red_intensity - np.mean(train_properties['Mean Red Relative Intensity'])) > drift_threshold:
                drifted_images['Mean Red Relative Intensity'].append(test_image_paths[idx])

    elif property_name == 'Mean Green Relative Intensity':
        drift_threshold = np.std(train_properties['Mean Green Relative Intensity']) * 2
        for idx, green_intensity in enumerate(test_properties['Mean Green Relative Intensity']):
            if abs(green_intensity - np.mean(train_properties['Mean Green Relative Intensity'])) > drift_threshold:
                drifted_images['Mean Green Relative Intensity'].append(test_image_paths[idx])

    elif property_name == 'Mean Blue Relative Intensity':
        drift_threshold = np.std(train_properties['Mean Blue Relative Intensity']) * 2
        for idx, blue_intensity in enumerate(test_properties['Mean Blue Relative Intensity']):
            if abs(blue_intensity - np.mean(train_properties['Mean Blue Relative Intensity'])) > drift_threshold:
                drifted_images['Mean Blue Relative Intensity'].append(test_image_paths[idx])

    # Save drifted images for each property in separate subfolders
    property_outliers_dir = os.path.join(base_outliers_dir, property_name.replace(" ", "_"))
    os.makedirs(property_outliers_dir, exist_ok=True)

    for img_path in drifted_images[property_name]:
        if os.path.exists(img_path):
            dst_path = os.path.join(property_outliers_dir, os.path.basename(img_path))
            shutil.copy(img_path, dst_path)

# Save the paths of drifted images to CSV files for each property
for property_name, img_paths in drifted_images.items():
    property_df = pd.DataFrame({f'{property_name} Drifted Image Paths': img_paths})
    property_df.to_csv(f'/Users/ximenamoure/Desktop/drift_last/results_deepchecks/original_results/{property_name}_drifted_images.csv', index=False)

print(f"Drifted images identified and saved in separate folders for each property.")
