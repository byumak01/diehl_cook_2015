from brian2 import *
import numpy as np
import struct

def get_spiking_rates_and_labels(test_phase, image_count, seed_data, dataset_path: str = "mnist/"):
    name = 't10k' if test_phase else 'train'
    
    # Load the images and labels
    image_intensities = _load_images(dataset_path + f'{name}-images.idx3-ubyte')
    image_intensities = _convert_indices_to_1d(image_intensities)
    image_rates = _convert_to_spiking_rates(image_intensities)

    image_labels = _load_labels(dataset_path + f'{name}-labels.idx1-ubyte')

    # Check if image_count is greater than the available images
    num_images = image_rates.shape[0]
    if image_count > num_images:
        raise ValueError(f"Requested image_count {image_count} exceeds the number of available images {num_images}.")
    
    # Select random indices
    if seed_data:
        np.random.seed(42)
    random_indices = np.random.choice(num_images, size=image_count, replace=False)
    
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

# Normalizes pixel intensities between 0 and 63.75. Normalized value will
# be spiking rate (Hz) of the cell.
def _convert_to_spiking_rates(images):
    return (images * 63.75) / 255

# Converts indices spiking rates from 2d to 1d, so that it can be used in
# PoissonGroup object.
def _convert_indices_to_1d(images):
    return images.reshape(images.shape[0], -1)

def increase_spiking_rates(image, current_max_rate):
    new_maximum_rate = current_max_rate + 32
    return (image * new_maximum_rate) / current_max_rate

def divisive_weight_normalization(synapse: Synapses, population_exc: int) -> None:
    for post_idx in range(population_exc):
        # Extract indices of synapses that connect to the current post-synaptic neuron
        target_indices = np.where(synapse.j == post_idx)[0]

        # Extract weights of these synapses
        weights_to_same_post = synapse.w_ee[target_indices]

        # Calculate sum of weights connected to the current post-synaptic neuron
        sum_of_weights = np.sum(weights_to_same_post)

        # Calculate normalization factor
        normalization_factor = 78 / sum_of_weights
        
        # Update the weights in the Synapses object
        synapse.w_ee[target_indices] *= normalization_factor

def receptive_field_for(neuron_idx, rf_size):
    half_size = rf_size // 2
    return [28 * j + i for i, j in [(i, j) for i in range((neuron_idx % 28) - half_size, (neuron_idx % 28) + half_size + 1) 
                                    for j in range((neuron_idx // 28) - half_size, (neuron_idx // 28) + half_size + 1) 
                                    if 0 <= i < 28 and 0 <= j < 28]]

def synapse_connections(neuron_population, rf_size):
    return np.transpose([(x, i) for i in range(neuron_population) for x in receptive_field_for(i, rf_size)])
