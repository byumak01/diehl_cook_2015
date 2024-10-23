from brian2 import *
from datetime import datetime
import numpy as np
import struct, argparse, csv, os


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
        # magic, num = struct.unpack(">II", f.read(8))
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


def divisive_weight_normalization(model, synapse: Synapses) -> None:
    for post_idx in range(model.args.population_exc):
        # Extract indices of synapses that connect to the current post-synaptic neuron
        target_indices = np.where(synapse.j == post_idx)[0]

        # Extract weights of these synapses
        weights_to_same_post = synapse.w_ee[target_indices]

        # Calculate sum of weights connected to the current post-synaptic neuron
        sum_of_weights = np.sum(weights_to_same_post)

        # Calculate normalization factor
        normalization_factor = model.args.normalization_const / sum_of_weights

        # Update the weights in the Synapses object
        synapse.w_ee[target_indices] *= normalization_factor


def receptive_field_for_exc(model, neuron_idx):
    half_size = model.args.rf_size // 2
    return [model.layout * j + i for j in
            range((neuron_idx // model.layout) - half_size, (neuron_idx // model.layout) + half_size + 1) for
            i in range((neuron_idx % model.layout) - half_size, (neuron_idx % model.layout) + half_size + 1)
            if 0 <= i < model.layout and 0 <= j < model.layout]


def receptive_field_for_inh(model, neuron_idx):
    half_size = model.args.rf_size // 2
    return [model.layout * j + i for j in
            range((neuron_idx // model.layout) - half_size, (neuron_idx // model.layout) + half_size + 1) for
            i in range((neuron_idx % model.layout) - half_size, (neuron_idx % model.layout) + half_size + 1)
            if 0 <= i < model.layout and 0 <= j < model.layout and model.layout * j + i != neuron_idx]


def synapse_connections_exc(model):
    return np.transpose([(x, i) for i in range(model.args.population_exc) for x in receptive_field_for_exc(model, i)])


def synapse_connections_inh(model):
    return np.transpose([(x, i) for i in range(model.args.population_inh) for x in receptive_field_for_inh(model, i)])


def draw_heatmap(model, spike_counts, img_name):
    spike_counts_grid = spike_counts.reshape(model.layout, model.layout)

    plt.clf()
    # Plotting the spike counts in a grid
    plt.figure(figsize=(12, 12))
    plt.imshow(spike_counts_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Spike Count')
    plt.title(f'{img_name}')
    plt.xlabel('Neuron X')
    plt.ylabel('Neuron Y')

    # Optional: annotate each square with the spike count
    for i in range(model.layout):
        for j in range(model.layout):
            plt.text(j, i, int(spike_counts_grid[i, j]), ha='center', va='center', color='white')
    plt.savefig(f"{model.run_path}/{img_name}.png")
    # plt.show()


def draw_weights(model, synapse, img_name):
    fig, ax = plt.subplots(model.layout, model.layout, figsize=(40, 40))
    dim = int(sqrt(model.args.population_exc))
    for post_idx in range(model.args.population_exc):
        pre_indices_for_current_post = sort(receptive_field_for_exc(model, post_idx))
        weights = np.zeros((dim, dim), dtype=float)
        weights[pre_indices_for_current_post//28, pre_indices_for_current_post%28] = synapse.w_ee[pre_indices_for_current_post, post_idx]

        row = post_idx // model.layout
        col = post_idx % model.layout

        ax[row, col].imshow(weights, vmin=0, vmax=1)
        ax[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(f"{model.run_path}/{img_name}.png")


def draw_accuracies(model, accuracies):
    if model.args.test_phase:
        run_label = "test"
    else:
        run_label = "training"
    # iteration is x label of graph
    iteration = [run_cnt * model.args.image_count + img_idx for run_cnt in range(model.args.run_count) for img_idx in
                 range(model.args.acc_update_interval, model.args.image_count + 1, model.args.acc_update_interval)]

    plt.figure(100)
    plt.plot(iteration, accuracies)
    plt.title(f'Accuracy change over iterations for {run_label} phase')
    plt.xlabel("Iteration Count")
    plt.ylabel("Accuracy % ")
    plt.grid(True)
    plt.savefig(f'{model.run_path}/{run_label}_accuracy_graph.png')


def check_update(curr_image_idx, update_interval):
    if curr_image_idx % update_interval == 0 and curr_image_idx != 0:
        return True
    else:
        return False


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
    parser.add_argument('--E_exc_for_inh', type=float, default=0,
                        help="Excitatory reversal potential for inhibitory neurons (mV)")
    parser.add_argument('--E_inh_for_inh', type=float, default=-85,
                        help="Inhibitory reversal potential for inhibitory neurons (mV)")
    parser.add_argument('--tau_lif_exc', type=float, default=100, help="LIF decay for excitatory neurons (ms)")
    parser.add_argument('--tau_lif_inh', type=float, default=10, help="LIF decay for inhibitory neurons (ms)")
    parser.add_argument('--tau_ge', type=float, default=1, help="Excitatory conductance decay (ms)")
    parser.add_argument('--tau_gi', type=float, default=2, help="Inhibitory conductance decay (ms)")
    parser.add_argument('--tau_theta', type=float, default=1e7, help="Theta decay rate (ms)")
    parser.add_argument('--theta_inc_exc', type=float, default=0.05, help="Theta increment for excitatory neurons (mV)")
    parser.add_argument('--refractory_exc', type=float, default=5, help="Refractory period for excitatory neurons (ms)")
    parser.add_argument('--refractory_inh', type=float, default=2, help="Refractory period for inhibitory neurons (ms)")
    parser.add_argument('--v_threshold_exc', type=float, default=-52,
                        help="Spiking threshold for excitatory neurons (mV)")
    parser.add_argument('--v_threshold_inh', type=float, default=-40,
                        help="Spiking threshold for inhibitory neurons (mV)")
    parser.add_argument('--v_offset_exc', type=float, default=20, help="Threshold offset for excitatory neurons (mV)")
    parser.add_argument('--v_reset_exc', type=float, default=-65, help="Reset voltage for excitatory neurons (mV)")
    parser.add_argument('--v_reset_inh', type=float, default=-45, help="Reset voltage for inhibitory neurons (mV)")
    parser.add_argument('--population_exc', type=int, default=784, help="Population of excitatory neurons")
    parser.add_argument('--population_inh', type=int, default=784, help="Population of inhibitory neurons")

    # Add Synapse parameters
    parser.add_argument('--tau_Apre_ee', type=float, default=20, help="Apre decay for exc.->exc. synapse (ms)")
    parser.add_argument('--tau_Apost1_ee', type=float, default=20, help="Apost1 decay for exc.->exc. synapse (ms)")
    parser.add_argument('--tau_Apost2_ee', type=float, default=40, help="Apost2 decay for exc.->exc. synapse (ms)")
    parser.add_argument('--eta_pre_ee', type=float, default=0.0001,
                        help="Pre-synaptic learning rate for exc.->exc. synapse")
    parser.add_argument('--eta_post_ee', type=float, default=0.01,
                        help="Post-synaptic learning rate for exc.->exc. synapse")
    parser.add_argument('--w_min_ee', type=float, default=0, help="Minimum weight for exc.->exc. synapse")
    parser.add_argument('--w_max_ee', type=float, default=1, help="Maximum weight for exc.->exc. synapse")
    parser.add_argument('--w_ei_', type=float, default=10.4, help="Weight for exc.->inh. synapse")
    parser.add_argument('--w_ie_', type=float, default=17, help="Weight for inh.->exc. synapse")
    parser.add_argument('--delay_ee', type=float, default=10, help="Delay for exc.->exc. synapse (ms)")
    parser.add_argument('--g_e_multiplier', type=float, default=1,
                        help="g_e_multiplier (on_pre -> g_e_post = w_ee * g_e_multiplier)")
    parser.add_argument('--normalization_const', type=float, default=78,
                        help="Normalization constant for div. w. norm.")
    # Add PoissonGroup parameters
    parser.add_argument('--max_rate', type=float, default=63.75, help="Maximum rate for PoissonGroup (Hz)")
    # Other params
    parser.add_argument('--seed_data', action='store_true', help="Set this flag to seed the data")
    parser.add_argument('--rf_size', type=int, default=3, help="RF size of neurons")
    parser.add_argument('--test_phase', action='store_true', help="Set this flag to indicate test_phase")
    parser.add_argument('--run_count', type=int, default=1, help="How many times dataset will be iterated")
    parser.add_argument('--image_count', type=int, default=5000, help="How many images will be used for run")
    parser.add_argument('--draw_update_interval', type=int, default=500, help="Update interval for heatmaps")
    parser.add_argument('--acc_update_interval', type=int, default=500, help="Update interval for accuracy")
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%m%d_%H%M%S"), help="run name")

    return parser.parse_args()


def write_to_csv(model, accuracy, sim_time, filename='runs.csv'):
    # Get a dictionary of all arguments
    args_dict = vars(model.args)

    # Add the accuracy to the dictionary
    args_dict['accuracy'] = accuracy
    args_dict['sim_time'] = sim_time

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
