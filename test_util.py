from brian2 import *
import numpy as np
import struct, argparse, csv, os

def get_spiking_rates_and_labels(test_phase, image_count, seed_data, max_rate, dataset_path: str = "mnist/"):
    name = 't10k' if test_phase else 'train'
    
    # Load the images and labels
    image_intensities = _load_images(dataset_path + f'{name}-images.idx3-ubyte')
    image_intensities = _convert_indices_to_1d(image_intensities)
    image_rates = _convert_to_spiking_rates(image_intensities, max_rate)

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

# Normalizes pixel intensities between 0 and max_rate. Normalized value will
# be spiking rate (Hz) of the cell.
def _convert_to_spiking_rates(images, max_rate):
    return (images * max_rate) / 255

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

def receptive_field_for_exc(neuron_idx, rf_size):
    half_size = rf_size // 2
    return [28 * j + i for i, j in [(i, j) for i in range((neuron_idx % 28) - half_size, (neuron_idx % 28) + half_size + 1) 
                                    for j in range((neuron_idx // 28) - half_size, (neuron_idx // 28) + half_size + 1) 
                                    if 0 <= i < 28 and 0 <= j < 28]]


def receptive_field_for_inh(neuron_idx, rf_size):
    half_size = rf_size // 2
    return [28 * j + i for i, j in [(i, j) for i in range((neuron_idx % 28) - half_size, (neuron_idx % 28) + half_size + 1) 
                                    for j in range((neuron_idx // 28) - half_size, (neuron_idx // 28) + half_size + 1) 
                                    if 0 <= i < 28 and 0 <= j < 28 and 28 * j + i != neuron_idx]]

def synapse_connections_exc(neuron_population, rf_size):
    return np.transpose([(x, i) for i in range(neuron_population) for x in receptive_field_for_exc(i, rf_size)])

def synapse_connections_inh(neuron_population, rf_size):
    return np.transpose([(x, i) for i in range(neuron_population) for x in receptive_field_for_inh(i, rf_size)])

def draw_heatmap(spike_counts, path, img_name):
    # Reshape the spike counts to a 28x28 grid
    spike_counts_grid = spike_counts.reshape(28, 28)

    plt.clf()
    # Plotting the spike counts in a grid
    plt.figure(figsize=(12,12))
    plt.imshow(spike_counts_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Spike Count')
    plt.title(f'{img_name}')
    plt.xlabel('Neuron X')
    plt.ylabel('Neuron Y')

    # Optional: annotate each square with the spike count
    for i in range(28):
        for j in range(28):
            plt.text(j, i, int(spike_counts_grid[i, j]), ha='center', va='center', color='white')
    plt.savefig(f"{path}/{img_name}.png")
    #plt.show()

def get_args():
    parser = argparse.ArgumentParser(
            description="Neuron, Synapse, and PoissonGroup parameters",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )


    # Add NeuronGroup parameters
    parser.add_argument('--E_rest_exc', type=float, default=-65, help="Resting potential for excitatory neurons (mV)")
    parser.add_argument('--E_rest_inh', type=float, default=-60, help="Resting potential for inhibitory neurons (mV)")
    parser.add_argument('--E_exc_for_exc', type=float, default=0, help="Excitatory reversal potential (mV)")
    parser.add_argument('--E_inh_for_exc', type=float, default=-100, help="Inhibitory reversal potential (mV)")
    parser.add_argument('--E_exc_for_inh', type=float, default=0, help="Excitatory reversal potential for inhibitory neurons (mV)")
    parser.add_argument('--E_inh_for_inh', type=float, default=-85, help="Inhibitory reversal potential for inhibitory neurons (mV)")
    parser.add_argument('--tau_lif_exc', type=float, default=100, help="LIF decay for excitatory neurons (ms)")
    parser.add_argument('--tau_lif_inh', type=float, default=10, help="LIF decay for inhibitory neurons (ms)")
    parser.add_argument('--tau_ge', type=float, default=1, help="Excitatory conductance decay (ms)")
    parser.add_argument('--tau_gi', type=float, default=2, help="Inhibitory conductance decay (ms)")
    parser.add_argument('--tau_theta', type=float, default=1e7, help="Theta decay rate (ms)")
    parser.add_argument('--theta_inc_exc', type=float, default=0.05, help="Theta increment for excitatory neurons (mV)")
    parser.add_argument('--refractory_exc', type=float, default=5, help="Refractory period for excitatory neurons (ms)")
    parser.add_argument('--refractory_inh', type=float, default=2, help="Refractory period for inhibitory neurons (ms)")
    parser.add_argument('--v_threshold_exc', type=float, default=-52, help="Spiking threshold for excitatory neurons (mV)")
    parser.add_argument('--v_threshold_inh', type=float, default=-40, help="Spiking threshold for inhibitory neurons (mV)")
    parser.add_argument('--v_offset_exc', type=float, default=20, help="Threshold offset for excitatory neurons (mV)")
    parser.add_argument('--v_reset_exc', type=float, default=-65, help="Reset voltage for excitatory neurons (mV)")
    parser.add_argument('--v_reset_inh', type=float, default=-45, help="Reset voltage for inhibitory neurons (mV)")
    parser.add_argument('--population_exc', type=int, default=784, help="Population of excitatory neurons")
    parser.add_argument('--population_inh', type=int, default=784, help="Population of inhibitory neurons")

    # Add Synapse parameters
    parser.add_argument('--tau_Apre_ee', type=float, default=20, help="Apre decay for exc.->exc. synapse (ms)")
    parser.add_argument('--tau_Apost1_ee', type=float, default=20, help="Apost1 decay for exc.->exc. synapse (ms)")
    parser.add_argument('--tau_Apost2_ee', type=float, default=40, help="Apost2 decay for exc.->exc. synapse (ms)")
    parser.add_argument('--eta_pre_ee', type=float, default=0.0001, help="Pre-synaptic learning rate for exc.->exc. synapse")
    parser.add_argument('--eta_post_ee', type=float, default=0.01, help="Post-synaptic learning rate for exc.->exc. synapse")
    parser.add_argument('--w_min_ee', type=float, default=0, help="Minimum weight for exc.->exc. synapse")
    parser.add_argument('--w_max_ee', type=float, default=1, help="Maximum weight for exc.->exc. synapse")
    parser.add_argument('--w_ei_', type=float, default=10.4, help="Weight for exc.->inh. synapse")
    parser.add_argument('--w_ie_', type=float, default=17, help="Weight for inh.->exc. synapse")
    parser.add_argument('--delay_ee', type=float, default=10, help="Delay for exc.->exc. synapse (ms)")
    parser.add_argument('--g_e_multiplier', type=float, default=1, help="g_e_multiplier (on_pre -> g_e_post = w_ee * g_e_multiplier)")
    # Add PoissonGroup parameters
    parser.add_argument('--max_rate', type=float, default=63.75, help="Maximum rate for PoissonGroup (Hz)")
    # Other params
    parser.add_argument('--seed_data', action='store_true', help="Set this flag to seed the data")
    parser.add_argument('--rf_size', type=int, default=3, help="RF size of neurons")
    parser.add_argument('--test_phase', action='store_true', help="Set this flag to indicate test_phase")
    parser.add_argument('--run_count', type=int, default=1, help="How many times dataset will be iterated")
    parser.add_argument('--image_count', type=int, default=5000, help="How many images will be used for run")
    parser.add_argument('--update_interval', type=int, default=500, help="Update interval for accuracy and heatmaps")
#
    return parser.parse_args()

def write_to_csv(args, accuracy, run_name, filename='runs.csv'):
    # Get a dictionary of all arguments
    args_dict = vars(args)

    # Add the accuracy to the dictionary
    args_dict['accuracy'] = accuracy
    args_dict['run_name'] = run_name

    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)

    # Writing to a CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header (parameter names) if file does not exist
        if not file_exists:
            writer.writerow(args_dict.keys())

        # Write parameter values (append to the file)
        writer.writerow(args_dict.values())

    print(f"Parameters and accuracy appended to {filename}")

