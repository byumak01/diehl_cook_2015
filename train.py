# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:57:54 2024

@author: ege-demir
"""

from functions import *
from norm import *
import argparse
import brian2
import numpy as np
import os
import shutil
from datetime import datetime

#TRAINNG CODE WITH LABELING PRED ACC
if 'i' in globals():
    del i
if 'j' in globals():
    del j

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--Connection", help = "Connection scheme (fc or lc)", type=str)
parser.add_argument("-s", "--Seed", help= "Seed number, should be an integer", type=int)
parser.add_argument("-fs", "--FilterSize", help= "Filter size, should be an integer", type=int)
parser.add_argument("-p", "--Padding", help= "Padding size, should be an integer", type=int)
args = parser.parse_args()

#run config
timestamp = datetime.now().strftime('%m%d-%H%M%S')
run_dir = f"{timestamp}-{args.Connection}-{args.Seed}"
os.makedirs(f"{run_dir}", exist_ok=True)

image_count = 60
exc_neuron_num = 400
tr_label_pred = True
label_predict_range = 10
size_selected = 28
max_rate = 63.75
seed_train = True
seed_train_val = args.Seed 

normalization_val = round(get_norm_dict()[313600])  # returns 78
print(f"[INFO] norm val: {normalization_val}")

lc = True

if args.Connection == "fc":
    lc = False
elif args.Connection == "lc":
    lc = True
else:
    raise Exception("Connection should be either fc or lc")

if lc:
    directory_of_mapping = 'input_to_output_mapping/refined_rf_trials'
    input_size = 28
    filter_size = args.FilterSize
    stride = 1
    padding = args.Padding
    output_size = 20
    mapping_file = f"{directory_of_mapping}/output_to_input_mapping_{input_size}_" \
                   f"{filter_size}_{stride}_{padding}_" \
                   f"{output_size}.npz"

    # Load mapping data
    mapping_data = np.load(f"{mapping_file}")
    print(f"[INFO] Loaded mapping data from '{mapping_file}'")

    pre_neuron_idx_input = mapping_data['input_neuron_idx']
    post_neuron_idx_exc = mapping_data['output_neuron_idx']

    print(f"[INFO] input_neuron_idx len '{len(pre_neuron_idx_input)}'")
    print(f"[INFO] output_neuron_idx len '{len(post_neuron_idx_exc)}'")


# Define path to save run info
run_info_path = os.path.join(run_dir, "run_info.txt")

# Compose run info as text
run_info_lines = [
    f"Run directory: {run_dir}",
    f"Image count: {image_count}",
    f"Excitatory neurons: {exc_neuron_num}",
    f"Training with label prediction: {tr_label_pred}",
    f"Label prediction range: {label_predict_range}",
    f"Image size: {size_selected}",
    f"Max firing rate: {max_rate}",
    f"Seeded training: {seed_train}",
    f"Seed: {seed_train_val}",
    f"Normalization value: {normalization_val}",
    f"Locally Connected: {lc}",
]

if lc:
    run_info_lines.extend([
        f"Input size: {input_size}",
        f"Filter size: {filter_size}",
        f"Stride: {stride}",
        f"Padding: {padding}",
        f"Output size: {output_size}",
        f"Mapping file: {mapping_file}",
        f"Input neuron indices: {len(pre_neuron_idx_input)}",
        f"Output neuron indices: {len(post_neuron_idx_exc)}",
    ])

# Save to file
with open(run_info_path, 'w') as f:
    f.write("\n".join(run_info_lines))

print(f"[INFO] Run info saved to {run_info_path}")

os.makedirs(f"{run_dir}/tr_label_pred", exist_ok=True)

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
tau_apre_ee = 20 * brian2.ms
tau_apost1_ee = 20 * brian2.ms
eta_pre_ee  = 0.0001
eta_post_ee = 0.01
w_min_ee = 0
w_max_ee = 1
w_ei_ = 10.4
w_ie_ = 17
tau_apost2_ee = 40 * brian2.ms  # Time constant for apost2_ee


# Neuron equations for excitatory and inhibitory populations
ng_eqs_exc = """
dv/dt = ((E_rest_exc - v) + g_e*(E_exc_for_exc - v) + g_i*(E_inh_for_exc - v))/tau_lif_exc : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
dtheta/dt = -theta/tau_theta  : volt
"""

ng_eqs_inh = """
dv/dt = ((E_rest_inh - v) + g_e*(E_exc_for_inh - v) + g_i*(E_inh_for_inh - v))/tau_lif_inh : volt (unless refractory)
dg_e/dt = -g_e/tau_ge : 1
dg_i/dt = -g_i/tau_gi : 1
"""


# Threshold and reset equations for both populations
ng_threshold_exc = "v > v_threshold_exc - v_offset_exc + theta"
ng_reset_exc = "v = v_reset_exc; theta += theta_inc_exc"

ng_threshold_inh = "v > v_threshold_inh"
ng_reset_inh = "v = v_reset_inh"

syn_eqs_ee_training = """
w_ee : 1
apost2_prev_ee : 1
dapre_ee/dt = -apre_ee/tau_apre_ee : 1 (event-driven)
dapost1_ee/dt = -apost1_ee/tau_apost1_ee : 1 (event-driven)
dapost2_ee/dt = -apost2_ee/tau_apost2_ee : 1 (event-driven)
"""

syn_on_pre_ee_training = """
apre_ee = 1
w_ee = clip(w_ee + (-eta_pre_ee * apost1_ee), w_min_ee, w_max_ee)
g_e_post += w_ee
"""

syn_on_post_ee_training = """
apost2_prev_ee = apost2_ee
w_ee = clip(w_ee + (eta_post_ee * apre_ee * apost2_prev_ee), w_min_ee, w_max_ee)
apost1_ee = 1
apost2_ee = 1
"""


# Create neuron groups for excitatory and inhibitory neurons
neuron_group_exc = brian2.NeuronGroup(N=population_exc, model=ng_eqs_exc, threshold=ng_threshold_exc, reset=ng_reset_exc, refractory=5*brian2.ms, method="euler")
neuron_group_inh = brian2.NeuronGroup(N=population_inh, model=ng_eqs_inh, threshold=ng_threshold_inh, reset=ng_reset_inh, refractory=2*brian2.ms, method="euler")


# Set initial values
neuron_group_exc.v = E_rest_exc - 40 * brian2.mV
neuron_group_inh.v = E_rest_inh - 40 * brian2.mV
neuron_group_exc.theta = 20 * brian2.mV

# Define PoissonGroup for input image (MNIST)
tot_input_num = size_selected * size_selected
image_input = brian2.PoissonGroup(N=tot_input_num, rates=0*brian2.Hz)  # Will set the rates based on the image

# Synapse connecting input neurons to excitatory neurons
syn_input_exc = brian2.Synapses(image_input, neuron_group_exc, model=syn_eqs_ee_training, on_pre=syn_on_pre_ee_training, on_post=syn_on_post_ee_training, method="euler")

if lc:
    syn_input_exc.connect(i=pre_neuron_idx_input, j=post_neuron_idx_exc)
    print("[INFO] input2exc connected based on RF.")
else:
    syn_input_exc.connect()
    print("[INFO] input2exc connected all2all.")

# syn_input_exc.connect()



syn_input_exc.w_ee[:] = "rand() * 0.3"
syn_input_exc.delay = 10 * brian2.ms

# Synapse connecting excitatory -> inhibitory neurons (one-to-one)
syn_exc_inh = brian2.Synapses(neuron_group_exc, neuron_group_inh, model="w_ei : 1", on_pre="g_e_post += w_ei", method="euler")
syn_exc_inh.connect(j='i')  # One-to-one connection
syn_exc_inh.w_ei = w_ei_

# Synapse connecting inhibitory -> excitatory neurons (all-to-all except same index)
syn_inh_exc = brian2.Synapses(neuron_group_inh, neuron_group_exc, model="w_ie : 1", on_pre="g_i_post += w_ie", method="euler")
syn_inh_exc.connect("i != j")
syn_inh_exc.w_ie = w_ie_


#weight_mon = StateMonitor(syn_input_exc, 'w_ee', record=False)
spike_mon_exc = brian2.SpikeMonitor(neuron_group_exc)

net = brian2.Network(neuron_group_exc, neuron_group_inh, image_input, syn_input_exc, syn_exc_inh, syn_inh_exc, spike_mon_exc)
image_input_rates, image_labels = load_all_precomputed_images_and_labels(directory = "mnist-train", seed=seed_train_val)

all_spike_counts_per_image = []
current_max_rate = max_rate
predicted_labels = []
image_labels_in_loop = []
cumulative_accuracies = []
image_indexes_in_loop = []
spike_data_within_training = []

previous_spike_counts = np.zeros(population_exc, dtype=int)
total_retries = 0
total_seen_images_with_retries = 0


for image_index_in_loop in range(image_count):
    total_seen_images_with_retries += 1
    tot_seen_images = image_index_in_loop + 1
    if image_index_in_loop % 1000 == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] tot_seen_images: {tot_seen_images}, total_retries: {total_retries}, total_seen_images_with_retries: {total_seen_images_with_retries}")

    image_retries = 0
    successful_training = False

    while not successful_training:
        previous_spike_counts = spike_mon_exc.count[:]
        image_input.rates = image_input_rates[image_index_in_loop] * brian2.Hz

        for post_idx in range(population_exc):
            synapse = syn_input_exc
            target_indices = np.where(synapse.j == post_idx)[0]
            weights_to_same_post = synapse.w_ee[target_indices]
            sum_of_weights = np.sum(weights_to_same_post)
            normalization_factor = normalization_val / sum_of_weights
            synapse.w_ee[target_indices] *= normalization_factor

        net.run(350 * brian2.ms)

        current_spike_counts = spike_mon_exc.count[:]
        spike_counts_current_image = current_spike_counts - previous_spike_counts
        sum_spike_count = np.sum(spike_counts_current_image)

        if sum_spike_count >= 5:
            successful_training = True
            all_spike_counts_per_image.append(spike_counts_current_image.copy())

            if tr_label_pred and label_predict_range is not None and (tot_seen_images % label_predict_range == 0):
                # print("Training labeling and prediction: save labels and images for every label_predict_range images")
                start_index_train_labeling = tot_seen_images - label_predict_range # 0 for 10k tot seen images
                end_index_train_labeling = tot_seen_images # 10k for 10k tot seen images

                image_labels_within_training = image_labels[start_index_train_labeling:end_index_train_labeling] #[0:10k]
                spike_data_within_training = all_spike_counts_per_image[-label_predict_range:]

                spike_data_within_training_path = f'{run_dir}/tr_label_pred/spike_data_{int(start_index_train_labeling+1)}_{end_index_train_labeling}.npz'
                np.savez(spike_data_within_training_path, labels=image_labels_within_training, spike_counts=spike_data_within_training)

                spike_data_with_labels_folder_path = f'{run_dir}/tr_label_pred/spike_data_w_labels_{int(start_index_train_labeling+1)}_{end_index_train_labeling}'
                os.makedirs(spike_data_with_labels_folder_path, exist_ok=True)

                assigned_labels_within_training_path = load_and_assign_neuron_labels(selected_spike_data_path = spike_data_within_training_path,
                                                                                     spike_data_with_labels_folder_path = spike_data_with_labels_folder_path,
                                                                                     population_exc = population_exc)


            if tr_label_pred and label_predict_range is not None and (tot_seen_images > label_predict_range):  # e.g., 10001st image, 10002nd image etc
                start_index_train_predict = tot_seen_images - 1 #index is 10k, meaning start with 10001th image
                end_index_train_predict = tot_seen_images -1 + label_predict_range #index is 20k, meaning end with 20k th image (as [x:y] means include xth index but not yth index)

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

                if (tot_seen_images - 1) % label_predict_range == 0:

                    prediction_folder_within_training_path = f'{spike_data_with_labels_folder_path}/predictions_{int(start_index_train_predict+1)}_to_{end_index_train_predict}'
                    os.makedirs(prediction_folder_within_training_path, exist_ok=True)

                    predicted_labels_path = f'{prediction_folder_within_training_path}/predicted_labels.npy'
                    image_labels_in_loop_path = f'{prediction_folder_within_training_path}/image_labels_in_loop.npy'
                    image_indexes_in_loop_path = f'{prediction_folder_within_training_path}/image_indexes_in_loop.npy'
                    cumulative_accuracies_path = f'{prediction_folder_within_training_path}/cumulative_accuracies.npy'

                predicted_labels.append(predicted_label)
                image_labels_in_loop.append(true_label)
                image_indexes_in_loop.append(image_index_in_loop)

                # accuracy
                correct_predictions = np.sum(np.array(predicted_labels) == np.array(image_labels_in_loop))
                cumulative_accuracy = (correct_predictions / len(image_labels_in_loop)) * 100
                cumulative_accuracies.append(cumulative_accuracy)

                if tot_seen_images % label_predict_range == 0:
                    range_slice = slice(-label_predict_range, None)
                    np.save(predicted_labels_path, np.array(predicted_labels[range_slice]))
                    np.save(image_labels_in_loop_path, np.array(image_labels_in_loop[range_slice]))
                    np.save(image_indexes_in_loop_path, np.array(image_indexes_in_loop[range_slice]))
                    np.save(cumulative_accuracies_path, np.array(cumulative_accuracies[range_slice]))

            image_input.rates = 0 * brian2.Hz
            net.run(150 * brian2.ms)
            current_max_rate = max_rate
        else:
            image_retries += 1
            total_retries += 1
            new_maximum_rate = current_max_rate + 32
            image_input_rates[image_index_in_loop] = (image_input_rates[image_index_in_loop] * new_maximum_rate) / current_max_rate
            current_max_rate = new_maximum_rate
            image_input.rates = 0 * brian2.Hz
            net.run(150 * brian2.ms)


last_image_index = image_index_in_loop
save_simulation_state(run_dir,last_image_index,syn_input_exc,neuron_group_exc,neuron_group_inh)
all_spike_counts_per_image = np.array(all_spike_counts_per_image)
spike_counts_per_neuron_with_retries = spike_mon_exc.count[:]

spike_counts_per_neuron_without_retries = np.sum(all_spike_counts_per_image, axis=0)
np.save(f'{run_dir}/spike_counts_per_neuron_without_retries.npy', spike_counts_per_neuron_without_retries)
np.save(f'{run_dir}/spike_counts_per_neuron_with_retries.npy', spike_counts_per_neuron_with_retries)

# Save the spike data and labels for the current run
np.savez(f'{run_dir}/spike_data_with_labels.npz', labels=image_labels[:image_count], spike_counts=all_spike_counts_per_image)

print(f"[INFO] Spike data and counts saved for run: {run_dir}.")

if tr_label_pred and label_predict_range is not None:

    output_dir = f"{run_dir}/tr_label_pred"
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths for saving
    predicted_labels_path = f'{output_dir}/predicted_labels.npy'
    image_labels_in_loop_path = f'{output_dir}/image_labels_in_loop.npy'
    image_indexes_in_loop_path = f'{output_dir}/image_indexes_in_loop.npy'
    cumulative_accuracies_path = f'{output_dir}/cumulative_accuracies.npy'

    # Save the specified range of data
    np.save(predicted_labels_path, np.array(predicted_labels))
    np.save(image_labels_in_loop_path, np.array(image_labels_in_loop))
    np.save(image_indexes_in_loop_path, np.array(image_indexes_in_loop))
    np.save(cumulative_accuracies_path, np.array(cumulative_accuracies))

    finalize_prediction_report(output_dir,predicted_labels,image_labels_in_loop,cumulative_accuracies,image_indexes_in_loop)
    load_and_analyze_training_data(label_predict_range, output_dir)


colab_base_dir = '/content/drive/MyDrive/tubitak_colab'
target_dir = os.path.join(colab_base_dir, run_dir)

# If a previous run with the same name exists, raise a warning and stop
if os.path.exists(target_dir):
    raise FileExistsError(f"Target folder already exists: {target_dir}\n"
                          f"Choose a different name or delete the existing folder manually.")

