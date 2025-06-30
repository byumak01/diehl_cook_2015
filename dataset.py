import numpy as np
import os, struct
import matplotlib.pyplot as plt

# Define dataset paths
dataset_path = "mnist/"
train_dataset_path = "mnist-train"
test_dataset_path = "mnist-test"
# Ensure directories exist
for path in [train_dataset_path, test_dataset_path]:
    os.makedirs(path, exist_ok=True)

def _load_images(filename: str):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def _load_labels(filename: str):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Download MNIST dataset (train and test)
mnist_train_images = _load_images(dataset_path + f'train-images.idx3-ubyte') 
mnist_test_images = _load_images(dataset_path + f't10k-images.idx3-ubyte') 

mnist_train_labels =_load_labels(dataset_path + f'train-labels.idx1-ubyte') 
mnist_test_labels = _load_labels(dataset_path + f't10k-labels.idx1-ubyte') 


print(np.shape(mnist_train_images))
print(np.shape(mnist_test_images))

# Save full datasets
np.save(os.path.join(train_dataset_path, "images.npy"), mnist_train_images)
np.save(os.path.join(train_dataset_path, "labels.npy"), mnist_train_labels)
np.save(os.path.join(test_dataset_path, "images.npy"), mnist_test_images)
np.save(os.path.join(test_dataset_path, "labels.npy"), mnist_test_labels)


def convert_images_to_rates_shuffled(max_rate, directory, seed=42, shuffle=True):
    """
    Converts images in the directory to rate-based format. Optionally shuffles with a seed.

    Args:
        max_rate (float): Maximum firing rate for pixel scaling.
        directory (str): Directory containing 'images.npy' and 'labels.npy'.
        seed (int): Seed for reproducible shuffling (ignored if shuffle=False).
        shuffle (bool): Whether to shuffle images and labels before saving.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed (and optionally shuffled) images and labels.
    """
    # Paths
    images_path = os.path.join(directory, "images.npy")
    labels_path = os.path.join(directory, "labels.npy")

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing 'images.npy' or 'labels.npy' in {directory}.")

    # Load
    images = np.load(images_path).astype(np.float32)
    labels = np.load(labels_path)

    # Shuffle if needed
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(images.shape[0])
        images = images[indices]
        labels = labels[indices]
        suffix = f"_seed{seed}"
    else:
        suffix = ""  # no suffix for unshuffled data

    # Flatten and scale
    images_rates = (images.reshape(images.shape[0], -1) * max_rate) / 255

    # Save outputs
    np.save(os.path.join(directory, f"images_rates{suffix}.npy"), images_rates)
    np.save(os.path.join(directory, f"labels{suffix}.npy"), labels)

    print(f"Images saved as 'images_rates{suffix}.npy'")
    print(f"Labels saved as 'labels{suffix}.npy'")

    return images_rates, labels


for i in range(5):
    # Shuffled version with seed
    convert_images_to_rates_shuffled(max_rate=63.75, directory="mnist-train", seed=i, shuffle=True)
    
    # Shuffled version with seed
    convert_images_to_rates_shuffled(max_rate=63.75, directory="mnist-test", seed=i, shuffle=True)


# Unshuffled version (default order)
convert_images_to_rates_shuffled(max_rate=63.75, directory="mnist-train", shuffle=False)

# Unshuffled version (default order)
convert_images_to_rates_shuffled(max_rate=63.75, directory="mnist-test", shuffle=False)



