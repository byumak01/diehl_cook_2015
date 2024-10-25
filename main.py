"""
This script is a reproduction of the work presented in:

Original Research Article:
Peter U. Diehl, Matthew Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity," 
Frontiers in Computational Neuroscience, vol. 9, 2015. 
DOI: https://doi.org/10.3389/fncom.2015.00099

Original code available at: https://github.com/peter-u-diehl/stdp-mnist/tree/master

Rewritten by: Barış Yumak, 2024
"""
from test_util import (
    get_spiking_rates_and_labels,
    increase_spiking_rates,
    divisive_weight_normalization,
    synapse_connections_exc,
    synapse_connections_inh,
    check_update,
    draw_heatmap,
    draw_weights,
    draw_accuracies,
    write_to_csv
)
from test_evaluation import get_accuracy, acc_update
from test_model import Model
import os
from brian2 import *

model = Model()

if model.args.test_phase and (not os.path.exists(model.run_path) or not os.listdir(model.run_path)):
    raise ValueError(f"There isn't a run named {model.args.run_name} or folder is empty. Cannot run test phase.")

if not model.args.test_phase and not os.path.exists(f"{model.run_path}"):
    os.makedirs(model.run_path)
    print(f"Directory {model.run_path} created successfully.")

start = time.time()

# Creating NeuronGroup objects for exc. and inh. populations
neuron_group_exc = NeuronGroup(N=model.args.population_exc, model=model.ng_eqs_exc, threshold=model.ng_threshold_exc,
                               reset=model.ng_reset_exc, refractory=model.args.refractory_exc * ms, method="euler")
neuron_group_inh = NeuronGroup(N=model.args.population_inh, model=model.ng_eqs_inh, threshold=model.ng_threshold_inh,
                               reset=model.ng_reset_inh, refractory=model.args.refractory_inh * ms, method="euler")

# Setting initial values for exc. and inh. populations
model.set_ng_namespace(neuron_group_exc)
model.set_ng_namespace(neuron_group_inh)

model.exc_ng_initial_vals(neuron_group_exc)
model.inh_ng_initial_vals(neuron_group_inh)

syn_con_exc = synapse_connections_exc(model)
syn_con_inh = synapse_connections_inh(model)

# Creating Synapse object for exc. -> inh. connection
syn_exc_inh = Synapses(neuron_group_exc, neuron_group_inh, model=model.ei_syn_eqs, on_pre=model.ei_syn_on_pre,
                       method="euler")
syn_exc_inh.connect(j='i')  # One-to-one connection

model.set_syn_namespace(syn_exc_inh)
model.ei_syn_initial_vals(syn_exc_inh)

# Creating Synapse object for inh. -> exc. connection
syn_inh_exc = Synapses(neuron_group_inh, neuron_group_exc, model=model.ie_syn_eqs, on_pre=model.ie_syn_on_pre,
                       method="euler")
syn_inh_exc.connect(i=syn_con_inh[1], j=syn_con_inh[0])  # inh. neurons connected to all exc. neurons expect the one which has the same index

model.set_syn_namespace(syn_inh_exc)
model.ie_syn_initial_vals(syn_inh_exc)

# Defining PoissonGroup for inputs
image_input = PoissonGroup(N=784, rates=0 * Hz)  # rates are changed according to image later

# Creating synapse object for input -> exc. connection, since inputs neurons are also excitatory we use
# equations for exc. -> exc. (ee)
syn_input_exc = Synapses(image_input, neuron_group_exc, model=model.ee_syn_eqs, on_pre=model.ee_syn_on_pre,
                         on_post=model.ee_syn_on_post, method="euler")
syn_input_exc.connect(i=syn_con_exc[0], j=syn_con_exc[1])
# weight init function call

model.set_syn_namespace(syn_input_exc)
model.ee_syn_initial_vals(syn_input_exc)

# Defining SpikeMonitor to record spike counts of neuron in neuron_group_exc
spike_mon_ng_exc_temp = SpikeMonitor(neuron_group_exc, record=True)

# spike_mon_ng_exc = SpikeMonitor(neuron_group_exc, record=True)
# poisson_spike_mon = SpikeMonitor(image_input, record=True)
syn_input_exc_mon = StateMonitor(syn_input_exc, ['w_ee'], record=True, dt=2500 * 500 * ms)

# Getting spiking rates and labels according to run_mode
image_input_rates, image_labels = get_spiking_rates_and_labels(model)

run(0 * ms)

curr_image_idx = 0  # Tracks the index of the current image during iteration.

accuracies = []  # This list is for saving calculated accuracies during training.

spike_counts_per_image = []  # List to store the spike counts of each neuron for each image.
# First dimension represents image idx and second dimension shows spike counts.

max_rate_current_image = model.args.max_rate

for rc in range(model.args.run_count):
    while curr_image_idx < model.args.image_count:  # While loop which will continue until all training data is finished.
        if curr_image_idx % 50 == 0:
            print("----------------------------------")
            print(f"Current image: {curr_image_idx}")
            print(f"Elapsed time:", {time.time() - start})
            print("----------------------------------")
        image_input.rates = image_input_rates[curr_image_idx] * Hz  # Setting poisson neuron rates for current input image.

        divisive_weight_normalization(model, syn_input_exc)  # Apply weight normalization

        run(350 * ms)  # training network for 350 ms.

        spike_counts_current_image = spike_mon_ng_exc_temp.count[:]
        del spike_mon_ng_exc_temp
        spike_mon_ng_exc_temp = SpikeMonitor(neuron_group_exc, record=True)

        sum_spike_counts_current_image = sum(spike_counts_current_image)  # TODO: naming convention needs checking

        if sum_spike_counts_current_image < 5:
            # Input frequency for current image is increased by 32 Hz if sum of 
            # spike counts of all neurons for current image is smaller than 5 and
            # training is repeated.
            max_rate_current_image += 32
            image_input_rates[curr_image_idx] = increase_spiking_rates(image_input_rates[curr_image_idx],
                                                                       max_rate_current_image)

        else:
            accuracy_update = check_update(curr_image_idx, model.args.acc_update_interval)
            if accuracy_update:
                spike_counts_per_image = acc_update(model, curr_image_idx, image_labels, spike_counts_per_image, accuracies)

            draw_update = check_update(curr_image_idx, model.args.draw_update_interval)
            if draw_update:
                # draw_heatmap(spike_mon_ng_exc.count[:], f"{run_path}", f"R{run_count}_I{curr_image_idx}_exc1_spike")
                # draw_heatmap(poisson_spike_mon.count[:], f"{run_path}", f"R{run_count}_I{curr_image_idx}_poisson_spike")
                draw_weights(model, syn_input_exc, f"R{model.args.run_count}_I{curr_image_idx}_syn_input_weights")

            # add spike counts for current image
            spike_counts_per_image.append(spike_counts_current_image)

            # reset max_rate_current_image before presenting new image.
            max_rate_current_image = model.args.max_rate

            curr_image_idx += 1

        # Run simulation 150 ms without learning, before next image.
        image_input.rates = 0 * Hz
        run(150 * ms)

    print("----------------------------------")
    print(f"{rc + 1}. iteration over dataset is finished.")
    # Calculate accuracy after iteration over dataset is finished.
    image_labels_curr_interval = image_labels[model.args.image_count - model.args.acc_update_interval: model.args.image_count]

    accuracy = get_accuracy(model, spike_counts_per_image, image_labels_curr_interval)
    accuracies.append(accuracy)

    # Reset curr_image_idx and spike_counts_per_image before giving dataset again.
    curr_image_idx = 0
    spike_counts_per_image = []

end = time.time()
sim_time = end - start
print(f"Simulation time: {sim_time}")

if not model.args.test_phase:  # training phase
    # Save weights and theta values.
    weights = syn_input_exc.w_ee[:]
    np.save(f'{model.run_path}/input_to_exc_trained_weights.npy', weights)
    theta_values = neuron_group_exc.theta[:]
    np.save(f'{model.run_path}/theta_values_exc.npy', theta_values)

draw_accuracies(model, accuracies)
# draw_heatmap(model, spike_mon_ng_exc.count[:], "final_exc1_spikes")
# draw_heatmap(model, poisson_spike_mon.count[:], "final_poisson_spikes")

draw_weights(model, syn_input_exc, f"final_syn_input_weights")
write_to_csv(model, accuracies[-1], sim_time)
