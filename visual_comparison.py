import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract embeddings from images using ResNet50
def extract_image_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))  # ResNet50 expects 224x224 images
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess for ResNet50
    features = base_model.predict(img)
    return features.flatten()  # Flatten the output to a 1D array

# Function to extract features from all images in a dataset
def extract_features_from_dataset(dataset_dir):
    features = []
    for subfolder in os.listdir(dataset_dir):
        subfolder_path = os.path.join(dataset_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, filename)
                feature = extract_image_features(image_path)
                if feature is not None:
                    features.append(feature)
    return np.array(features)

# Function to perform PCA and visualize feature distributions
def visualize_pca(features_ref, features_new):
    pca = PCA(n_components=2)
    combined_features = np.vstack((features_ref, features_new))
    pca_result = pca.fit_transform(combined_features)
    
    # Split the PCA results back into reference and new datasets
    pca_ref = pca_result[:len(features_ref)]
    pca_new = pca_result[len(features_ref):]

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_ref[:, 0], pca_ref[:, 1], label='Reference Dataset', alpha=0.6)
    plt.scatter(pca_new[:, 0], pca_new[:, 1], label='New Dataset', alpha=0.6)
    plt.title('PCA of Image Features')
    plt.legend()
    plt.show()

# Main Function: Compare Reference and New Dataset using features
def main(reference_dir, new_dir):
    print("Extracting features from reference dataset...")
    reference_features = extract_features_from_dataset(reference_dir)
    
    print("Extracting features from new dataset...")
    new_features = extract_features_from_dataset(new_dir)
    
    print("Visualizing PCA comparison of features...")
    visualize_pca(reference_features, new_features)


if __name__ == "__main__":
    # Path to the reference and new (drifted) datasets
    reference_dataset = "/Users/ximenamoure/Desktop/drift_last/reference_dataset"
    new_dataset = "/Users/ximenamoure/Chess-Piece-Classification-Dataset/images/processed/occupancy/split0"

    # Run the comparison and visualization
    main(reference_dataset, new_dataset)
