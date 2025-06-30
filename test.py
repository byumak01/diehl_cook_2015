from collections import Counter
import os
import numpy as np
import math
from functions import *
from norm import *

#LABELING NEURONS FOR EVAL TESTING, USING SPIKE DATA AND IMAGE LABELS MONITORED IN TRAINING

# Labeling config
#labeling config is fixed for this basic code, you label the neurons using all the spiking data from the training
selected_spike_data_path = f'{run_dir}/spike_data_with_labels.npz'
spike_data_with_labels_folder_path = f'{run_dir}/regular_full'
os.makedirs(spike_data_with_labels_folder_path, exist_ok=True)

assigned_labels_path = load_and_assign_neuron_labels(selected_spike_data_path = selected_spike_data_path,
                                                     spike_data_with_labels_folder_path = spike_data_with_labels_folder_path,
                                                     population_exc = exc_neuron_num)


assigned_labels = np.load(assigned_labels_path)
print(f'[INFO] Loaded assigned labels from {assigned_labels_path}')
label_counts = Counter(assigned_labels)
save_folder = os.path.dirname(assigned_labels_path)
histogram_txt_path = os.path.join(save_folder, 'neuron_labels_histogram.txt')

with open(histogram_txt_path, 'w') as f:
    for label in range(-1, 10):
        count = label_counts.get(label, 0)
        f.write(f'Label {label}: {count}\n')
        print(f'Label {label}: {count}')

print(f'[INFO] Neuron label histogram data saved at {histogram_txt_path}')

print(f"Assigned labels:")
print(assigned_labels.reshape(int(math.sqrt(exc_neuron_num)), int(math.sqrt(exc_neuron_num))))

#pred config
image_count_prediction = 10
seed_test = True
seed_test_val = args.Seed

# Append prediction info to run_info.txt
prediction_info_lines = [
    "\n--- Prediction Phase ---",
    f"Prediction image count: {image_count_prediction}",
    f"Seeded test: {seed_test}",
    f"Seed (test): {seed_test_val}",
]

# Append to the same file
with open(run_info_path, 'a') as f:
    f.write("\n".join(prediction_info_lines))

print(f"[INFO] Prediction info appended to {run_info_path}")


folder_directory = os.path.dirname(assigned_labels_path)
folder_name = f"prediction_{image_count_prediction}"
prediction_folder_path = f'{folder_directory}/{folder_name}'

# Ensure the folder is created
os.makedirs(prediction_folder_path, exist_ok=True)

if 'i' in globals():
    del i
if 'j' in globals():
    del j

# Parameters for excitatory and inhibitory neurons
E_rest_exc  = -65 * brian2.mV
E_rest_inh  = -60 * brian2.mV
E_exc_for_exc = 0 * brian2.mV
E_inh_for_exc = -100 * brian2.mV
E_exc_for_inh = 0 * brian2.mV
E_inh_for_inh = -85 * brian2.mV
tau_lif_exc = 100 * brian2.ms
tau_lif_inh = 10 * brian2.ms
tau_ge  = 1 * brian2.ms
tau_gi  = 2 * brian2.ms
tau_theta =  1e7 * brian2.ms
theta_inc_exc =  0.05 * brian2.mV
v_threshold_exc = -52 * brian2.mV
v_threshold_inh = -40 * brian2.mV
v_offset_exc = 20 * brian2.mV
v_reset_exc = -65 * brian2.mV
v_reset_inh = -45 * brian2.mV
population_exc = exc_neuron_num  # Excitatory neuron population
population_inh = population_exc  # Inhibitory neuron population

# Synapse Parameters
tau_Apre_ee = 20 * brian2.ms
tau_Apost1_ee = 20 * brian2.ms
eta_pre_ee  = 0.0001
eta_post_ee = 0.01
w_min_ee = 0
w_max_ee = 1
w_ei_ = 10.4
w_ie_ = 17

# Neuron equations for excitatory and inhibitory populations
ng_eqs_exc = """
dv/dt = ((E_rest_exc - v) + g_e*(E_exc_for_exc - v) + g_i*(E_inh_for_exc - v))/tau_lif_exc : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
theta : volt
"""

ng_eqs_inh = """
dv/dt = ((E_rest_inh - v) + g_e*(E_exc_for_inh - v) + g_i*(E_inh_for_inh - v))/tau_lif_inh : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
"""

# Threshold and reset equations for both populations
ng_threshold_exc = "v > v_threshold_exc - v_offset_exc + theta"
ng_reset_exc = "v = v_reset_exc"

ng_threshold_inh = "v > v_threshold_inh"
ng_reset_inh = "v = v_reset_inh"


# Synapse equations for exc. -> exc. connections (test phase)
syn_eqs_ee_test = """
w_ee : 1
"""

syn_on_pre_ee_test = """
g_e_post += w_ee
"""

# Create neuron groups for excitatory and inhibitory neurons
neuron_group_exc = brian2.NeuronGroup(N=population_exc, model=ng_eqs_exc, threshold=ng_threshold_exc, reset=ng_reset_exc, refractory=5*brian2.ms, method="euler")
neuron_group_inh = brian2.NeuronGroup(N=population_inh, model=ng_eqs_inh, threshold=ng_threshold_inh, reset=ng_reset_inh, refractory=2*brian2.ms, method="euler")

# Set initial values
neuron_group_exc.v = E_rest_exc - 40 * brian2.mV
neuron_group_inh.v = E_rest_inh - 40 * brian2.mV #bu orijinal kodda neden commentli?

theta_values = np.load(f"{run_dir}/neuron_group_exc_theta.npy")
neuron_group_exc.theta = theta_values * brian2.volt


# Define PoissonGroup for input image (MNIST)
tot_input_num = size_selected * size_selected
image_input = brian2.PoissonGroup(N=tot_input_num, rates=0*brian2.Hz)  # Will set the rates based on the image

syn_input_exc = brian2.Synapses(image_input, neuron_group_exc, model=syn_eqs_ee_test, on_pre=syn_on_pre_ee_test, method="euler")
if lc:
    syn_input_exc.connect(i=pre_neuron_idx_input, j=post_neuron_idx_exc)
    print("[INFO] input2exc connected based on RF.")
else:
    syn_input_exc.connect()
    print("[INFO] input2exc connected all2all.")

# syn_input_exc.connect()


saved_weights = np.load(f'{run_dir}/final_synaptic_weights.npy')
syn_input_exc.w_ee[:] = saved_weights
print(f'[INFO] Loaded saved synaptic weights from run name {run_dir}.')

syn_input_exc.delay = 10 * brian2.ms

# Synapse connecting excitatory -> inhibitory neurons (one-to-one)
syn_exc_inh = brian2.Synapses(neuron_group_exc, neuron_group_inh, model="w_ei : 1", on_pre="g_e_post += w_ei", method="euler")
syn_exc_inh.connect(j='i')  # One-to-one connection
syn_exc_inh.w_ei = w_ei_

# Synapse connecting inhibitory -> excitatory neurons (all-to-all except same index)
syn_inh_exc = brian2.Synapses(neuron_group_inh, neuron_group_exc, model="w_ie : 1", on_pre="g_i_post += w_ie", method="euler")
syn_inh_exc.connect("i != j")
syn_inh_exc.w_ie = w_ie_

# Create a SpikeMonitor to record the spikes of excitatory neurons
spike_mon_exc = brian2.SpikeMonitor(neuron_group_exc)

# Create a network to encapsulate all components
net = brian2.Network(neuron_group_exc, neuron_group_inh, image_input, syn_input_exc, syn_exc_inh, syn_inh_exc, spike_mon_exc)

# Reset the network (reset all components to initial values)
net.store('initialized')  # Store the initialized state for later reset
image_input_rates, image_labels = load_all_precomputed_images_and_labels(directory = "mnist-test", seed = seed_test_val)

# Initialize a list to store spike counts for each image
all_spike_counts_per_image = []
predicted_labels = []
image_labels_in_loop = []
cumulative_accuracies = []
image_indexes_in_loop = []
max_rate_current_image = max_rate  # Set the initial maximum rate

previous_spike_counts = np.zeros_like(spike_mon_exc.count[:])  # Initialize previous spike counts to zero
max_rate_current_image = max_rate  # Set initial rate

for image_index_in_loop in range(image_count_prediction):
    total_seen_images_with_retries += 1
    tot_seen_images = image_index_in_loop + 1
    if image_index_in_loop % 1000 == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] tot_seen_images: {tot_seen_images}, total_retries: {total_retries}, total_seen_images_with_retries: {total_seen_images_with_retries}")

    image_retries = 0  # Track retries per image
    successful_test = False

    while not successful_test:
        previous_spike_counts = spike_mon_exc.count[:]
        image_input.rates = image_input_rates[image_index_in_loop] * brian2.Hz

        # Run the network for 350 ms
        net.run(350 * brian2.ms)

        # Get the current total spike counts after processing this image
        current_spike_counts = spike_mon_exc.count[:]
        spike_counts_current_image = current_spike_counts - previous_spike_counts  # Get spikes only for this image
        sum_spike_count = np.sum(spike_counts_current_image)

        # Check if any neuron spiked 5 or more times
        if sum_spike_count >= 5:
            successful_test = True
            all_spike_counts_per_image.append(spike_counts_current_image.copy())  # Copy current spikes to avoid overwriting later

            # predictions
            assigned_labels = np.load(assigned_labels_within_training_path)
            predictions_all_labels = []
            for label in range(10):
                assignment_indices = np.where(assigned_labels == label)[0]
                if len(assignment_indices) > 0:
                    total_spike_count = np.sum(spike_counts_current_image[assignment_indices])
                    average_spike_count = total_spike_count / len(assignment_indices)
                    predictions_all_labels.append(average_spike_count)
                else:
                    predictions_all_labels.append(0)

            predictions_all_labels_dec= list(np.argsort(predictions_all_labels)[::-1]) #decending
            true_label = image_labels[image_index_in_loop]
            predicted_label = predictions_all_labels_dec[0]

            # Append current labels and index to lists
            predicted_labels.append(predicted_label)
            image_labels_in_loop.append(true_label)
            image_indexes_in_loop.append(image_index_in_loop)

            # Convert lists to numpy arrays for accuracy calculation
            predicted_labels_np = np.array(predicted_labels)
            image_labels_in_loop_np = np.array(image_labels_in_loop)

            # Calculate cumulative accuracy
            correct_predictions = np.sum(predicted_labels_np == image_labels_in_loop_np)
            cumulative_accuracy = (correct_predictions / len(image_labels_in_loop_np)) * 100

            print(f"image_index_in_loop: {image_index_in_loop}, true_label: {true_label}, predicted_label: {predicted_label}, cumulative_accuracy: {cumulative_accuracy}")
            # Append to cumulative accuracies and save
            cumulative_accuracies.append(cumulative_accuracy)

            # Reset max_rate_current_image before presenting new image.
            max_rate_current_image = max_rate  # Reset to the original max_rate after successful image processing
            # Stop the input for 150 ms
            image_input.rates = 0 * brian2.Hz
            net.run(150 * brian2.ms)  # Simulate the network without input for 100 ms
        else:
            image_retries += 1
            total_retries += 1
            new_maximum_rate = current_max_rate + 32
            image_input_rates[image_index_in_loop] = (image_input_rates[image_index_in_loop] * new_maximum_rate) / current_max_rate
            current_max_rate = new_maximum_rate
            image_input.rates = 0 * brian2.Hz
            net.run(150 * brian2.ms)

predicted_labels_path = f'{prediction_folder_path}/predicted_labels.npy'
image_labels_in_loop_path = f'{prediction_folder_path}/image_labels_in_loop.npy'
image_indexes_in_loop_path = f'{prediction_folder_path}/image_indexes_in_loop.npy'

# Save the specified range of data
np.save(predicted_labels_path, np.array(predicted_labels))
np.save(image_labels_in_loop_path, np.array(image_labels_in_loop))
np.save(image_indexes_in_loop_path, np.array(image_indexes_in_loop))

cumulative_accuracies_path = f'{prediction_folder_path}/cumulative_accuracies.npy'
np.save(cumulative_accuracies_path, np.array(cumulative_accuracies))

print(f"[INFO] Test phase complete. Predictions made for {image_count_prediction} images with the UNSEEN TEST data.")

all_spike_counts_per_image = np.array(all_spike_counts_per_image)
spike_counts_per_neuron_with_retries = spike_mon_exc.count[:]

# Save spike counts without and with retries for current run
spike_counts_per_neuron_without_retries = np.sum(all_spike_counts_per_image, axis=0)
np.save(f'{prediction_folder_path}/spike_counts_per_neuron_without_retries.npy', spike_counts_per_neuron_without_retries)
np.save(f'{prediction_folder_path}/spike_counts_per_neuron_with_retries.npy', spike_counts_per_neuron_with_retries)
np.savez(f'{prediction_folder_path}/spike_data_with_labels.npz', labels=image_labels, spike_counts=all_spike_counts_per_image)

print(f"[INFO] Total spike data and counts saved for test : {prediction_folder_path}.")

accuracy = calculate_accuracy(prediction_folder_path = prediction_folder_path)
print(f"[INFO] Test Accuracy: {accuracy:.2f}%")

# Calculate accuracy per label
accuracy_per_label = calculate_accuracy_per_label(prediction_folder_path = prediction_folder_path)

