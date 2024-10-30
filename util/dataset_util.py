from brian2 import *
import struct

def get_spiking_rates_and_labels(model, dataset_path: str = "mnist/"):
    label = 't10k' if model.args.test_phase else 'train'

    # Load the images and labels
    image_intensities = _load_images(dataset_path + f'{label}-images.idx3-ubyte')
    image_intensities = _convert_indices_to_1d(image_intensities)
    image_rates = _convert_to_spiking_rates(image_intensities, model.args.max_rate)

    image_labels = _load_labels(dataset_path + f'{label}-labels.idx1-ubyte')

    # Check if image_count is greater than the available images
    num_images = image_rates.shape[0]
    if model.args.image_count > num_images:
        raise ValueError(
            f"Requested image_count {model.args.image_count} exceeds the number of available images {num_images}.")

    # Select random indices
    if model.args.seed_data:
        np.random.seed(42)
    random_indices = np.random.choice(num_images, size=model.args.image_count, replace=False)

    # Select the subset of images and labels
    image_rates_subset = image_rates[random_indices]
    image_labels_subset = image_labels[random_indices]

    return image_rates_subset, image_labels_subset


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


# Normalizes pixel intensities between 0 and max_rate. Normalized value will
# be spiking rate (Hz) of the cell.
def _convert_to_spiking_rates(images, max_rate):
    return (images * max_rate) / 255


# Converts indices spiking rates from 2d to 1d, so that it can be used in
# PoissonGroup object.
def _convert_indices_to_1d(images):
    return images.reshape(images.shape[0], -1)