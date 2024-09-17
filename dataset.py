"""
This script loads the MNIST dataset and converts pixel intensities 
into rate-encoded inputs, which are used for simulation.

Original author: Ege Demir, 2024

Modifications made by: Barış Yumak, 2024
"""
from brian2 import *
import numpy as np
import struct

def get_spiking_rates_and_labels(dataset_path: str = "mnist/"):
    # Load training data
    train_image_intensities = _load_images(dataset_path + 'train-images.idx3-ubyte')
    train_image_labels = _load_labels(dataset_path + 'train-labels.idx1-ubyte')
    print("Train image shape: ", train_image_intensities.shape)
    print("Train label shape: ", train_image_labels.shape)
    # Load test data
    test_image_intensities = _load_images(dataset_path + 't10k-images.idx3-ubyte')
    test_image_labels = _load_labels(dataset_path + 't10k-labels.idx1-ubyte')
    print("Test image shape: ", test_image_intensities.shape)
    print("Test label shape: ", test_image_labels.shape)

    # Convert 2d indices to 1d
    _train_image_intensities = _convert_indices_to_1d(train_image_intensities)
    _test_image_intensities = _convert_indices_to_1d(test_image_intensities)
    print("Train image shape after conversion: ", _train_image_intensities.shape)
    print("Test image shape after conversion: ", _test_image_intensities.shape)

    # Get spiking rates of images
    train_image_rates = _convert_to_spiking_rates(_train_image_intensities)
    test_image_rates = _convert_to_spiking_rates(_test_image_intensities)
    
    return train_image_rates, train_image_labels, test_image_rates, test_image_labels

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