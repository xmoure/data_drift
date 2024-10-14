import os
import numpy as np
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

SEED = 42

def load_occupancy_image_paths(data_folder):
    # List images
    occupied = glob(os.path.join(data_folder, "[!empty]*/*.png"))
    empty = glob(os.path.join(data_folder, "empty/*.png"))

    # Downsample the majority class
    rng = np.random.default_rng(seed=SEED)
    rng.shuffle(empty)
    empty = empty[:len(occupied)]

    print(f"Occupied samples: {len(occupied)}")
    print(f"Empty samples: {len(empty)}")

    # Combine paths
    paths = np.array(occupied + empty)
    return paths

# Define the path to your data folder
data_folder = "/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split5"
reference_folder = "/Users/ximenamoure/Desktop/drift_last/reference_dataset"
training_folder = "/Users/ximenamoure/Desktop/drift_last/training_dataset"

# Create directories for the datasets
os.makedirs(reference_folder, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)

# Load and downsample the data
paths = load_occupancy_image_paths(data_folder)

# Split the data
train_paths, reference_paths = train_test_split(paths, test_size=0.5, random_state=SEED)

# Function to copy files to the respective folders
def copy_files(file_paths, target_folder):
    for img_path in file_paths:
        piece = os.path.basename(os.path.dirname(img_path))
        piece_folder = os.path.join(target_folder, piece)
        os.makedirs(piece_folder, exist_ok=True)
        shutil.copy(img_path, piece_folder)

# Copy the files to the respective folders
print("Starting first copy...")
copy_files(reference_paths, reference_folder)
print("Starting second copy...")
copy_files(train_paths, training_folder)

print("Data split into reference and training datasets successfully.")
