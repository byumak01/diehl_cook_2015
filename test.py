"""
This script is a reproduction of the work presented in:

Original Research Article:
Peter U. Diehl, Matthew Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity," 
Frontiers in Computational Neuroscience, vol. 9, 2015. 
DOI: https://doi.org/10.3389/fncom.2015.00099

Original code available at: https://github.com/peter-u-diehl/stdp-mnist/tree/master

Rewritten by: Barış Yumak, 2024
"""
from datetime import datetime 
import time, os
from test_util import (
    get_spiking_rates_and_labels,
    increase_spiking_rates,
    divisive_weight_normalization,
    synapse_connections_exc,
    synapse_connections_inh,
    draw_heatmap,
    draw_weights,
    get_args,
    write_to_csv
    )
from test_evaluation import calculate_accuracy, get_predictions, assign_neurons_to_labels
# TODO: Needs check conditions to see whether image size and spike per image list length are equal

# Parse the arguments
args = get_args()

# Use the arguments
print(f'Test phase: {args.test_phase}')
print(f'Image count: {args.image_count}')
print(f'Update interval: {args.update_interval}')
print(f'Run count: {args.run_count}')

run_name = datetime.now().strftime("%m%d_%H%M%S")
run_path = f"runs/{run_name}"

from brian2 import * # importing this before input() creates conflict.

if args.test_phase and (not os.path.exists(run_path) or not os.listdir(run_path)):
    raise ValueError(f"There isn't a run named {run_name} or folder is empty. Cannot run test phase.")

if not args.test_phase and not os.path.exists(f"{run_path}"):
        os.makedirs(run_path)
        print(f"Directory {run_path} created successfully.")

start = time.time()

# Parameters
test_phase = args.test_phase
seed_data = args.seed_data
run_count = args.run_count
image_count = args.image_count
update_interval = args.update_interval
rf_size = args.rf_size
# NeuronGroup Parameters:
E_rest_exc = args.E_rest_exc * mV
E_rest_inh = args.E_rest_inh * mV
E_exc_for_exc = args.E_exc_for_exc * mV
E_inh_for_exc = args.E_inh_for_exc * mV
E_exc_for_inh = args.E_exc_for_inh * mV
E_inh_for_inh = args.E_inh_for_inh * mV
tau_lif_exc = args.tau_lif_exc * ms
tau_lif_inh = args.tau_lif_inh * ms
tau_ge = args.tau_ge * ms
tau_gi = args.tau_gi * ms
tau_theta = args.tau_theta * ms
theta_inc_exc = args.theta_inc_exc * mV
refractory_exc = args.refractory_exc * ms
refractory_inh = args.refractory_inh * ms
v_threshold_exc = args.v_threshold_exc * mV
v_threshold_inh = args.v_threshold_inh * mV
v_offset_exc = args.v_offset_exc * mV
v_reset_exc = args.v_reset_exc * mV
v_reset_inh = args.v_reset_inh * mV
population_exc = args.population_exc
population_inh = args.population_inh   
# Synapse Parameters:      
tau_Apre_ee = args.tau_Apre_ee * ms
tau_Apost1_ee = args.tau_Apost1_ee * ms
tau_Apost2_ee = args.tau_Apost2_ee * ms
eta_pre_ee = args.eta_pre_ee
eta_post_ee = args.eta_post_ee
w_min_ee = args.w_min_ee
w_max_ee = args.w_max_ee
w_ei_ = args.w_ei_
w_ie_ = args.w_ie_
delay_ee = args.delay_ee * ms
g_e_multiplier = args.g_e_multiplier
# PoissonGroup parameters:
max_rate = args.max_rate                                # Spike intensities are normalized between 0 and max_rate (Hz) at the beginning.

# NeuronGroup equations for exc. and inh. populations
ng_eqs_exc = """
dv/dt = ((E_rest_exc - v) + g_e*(E_exc_for_exc - v) + g_i*(E_inh_for_exc - v))/tau_lif_exc : volt (unless refractory)  # (Eq. 1 in the paper)
dg_e/dt = -g_e/tau_ge : 1                                                                                              # (Eq. 2 in the paper)
dg_i/dt = -g_i/tau_gi : 1                                                                                              # (Eq. 2 in the paper (Inhibitory version))
"""

# Theta is used for adaptive threshold mechanism. It uses its trained value in test phase.
# w_sum is used in divisive weight normalization.
if test_phase:
    ng_eqs_exc += "theta : volt"
else:
    ng_eqs_exc += "dtheta/dt = -theta/tau_theta  : volt"

ng_eqs_inh = """
dv/dt = ((E_rest_inh - v) + g_e*(E_exc_for_inh - v) + g_i*(E_inh_for_inh - v))/tau_lif_inh : volt (unless refractory)  # (Eq. 1 in the paper (inhibitory version))
dg_e/dt = -g_e/tau_ge : 1                                                                                              # (Eq. 2 in the paper)
dg_i/dt = -g_i/tau_gi : 1                                                                                              # (Eq. 2 in the paper (Inhibitory version))
"""

# Defining threshold equations for exc. and inh. populations
ng_threshold_exc = "v > v_threshold_exc - v_offset_exc + theta"
ng_threshold_inh = "v > v_threshold_inh"

# Defining reset equations for exc. and inh. populations
ng_reset_exc = """
v = v_reset_exc 
"""

if not test_phase:
    ng_reset_exc = "theta += theta_inc_exc"

ng_reset_inh = """
v = v_reset_inh
"""

# Synapse equations for exc. -> exc. connections (training phase)
syn_eqs_ee_training = """
w_ee : 1                                                     # w represents the weight
Apost2_prev_ee : 1                                           # holds previous value of A_post2
dApre_ee/dt = -Apre_ee/tau_Apre_ee : 1        (event-driven) # pre-synaptic trace
dApost1_ee/dt = -Apost1_ee/tau_Apost1_ee : 1  (event-driven) # post-synaptic trace 1
dApost2_ee/dt = -Apost2_ee/tau_Apost2_ee : 1  (event-driven) # post-synaptic trace 2
"""

syn_on_pre_ee_training = """
Apre_ee = 1
w_ee = clip(w_ee + (-eta_pre_ee * Apost1_ee), w_min_ee, w_max_ee)
g_e_post += w_ee * g_e_multiplier
"""

syn_on_post_ee_training = """
Apost2_prev_ee = Apost2_ee
w_ee = clip(w_ee + (eta_post_ee * Apre_ee * Apost2_prev_ee), w_min_ee, w_max_ee)
Apost1_ee = 1 
Apost2_ee = 1
"""

# Synapse equations for exc. -> exc. connections (test phase)
syn_eqs_ee_test = """
w_ee : 1                                                     # w represents the weight
"""

syn_on_pre_ee_test = """
g_e_post += w_ee
"""

# Synapse equations for exc. -> inh. connections
syn_eqs_ei = """
w_ei : 1
"""

syn_on_pre_ei = """
g_e_post += w_ei
"""

# Synapse equations for inh. -> exc. connections
syn_eqs_ie = """
w_ie : 1
"""

syn_on_pre_ie = """
g_i_post += w_ie
"""

# Creating NeuronGroup objects for exc. and inh. populations
neuron_group_exc = NeuronGroup(N=population_exc, model=ng_eqs_exc, threshold=ng_threshold_exc, reset=ng_reset_exc, refractory=refractory_exc, method="euler")
neuron_group_inh = NeuronGroup(N=population_inh, model=ng_eqs_inh, threshold=ng_threshold_inh, reset=ng_reset_inh, refractory=refractory_inh, method="euler")

# Setting initial values for exc. and inh. populations
neuron_group_exc.v = E_rest_exc - 40 * mV
neuron_group_inh.v = E_rest_inh - 40 * mV

if test_phase:
    theta_values_exc = np.load(f"{run_path}/theta_values_exc.npy")
    neuron_group_exc.theta = theta_values_exc * volt
else: # training phase
    neuron_group_exc.theta = 20 * mV

syn_con_exc = synapse_connections_exc(784, rf_size)
syn_con_inh = synapse_connections_inh(784, rf_size)

# Creating Synapse object for exc. -> inh. connection
syn_exc_inh = Synapses(neuron_group_exc, neuron_group_inh, model=syn_eqs_ei, on_pre=syn_on_pre_ei, method="euler")
syn_exc_inh.connect(j='i') # One-to-one connection

# Setting weight for exc. -> inh. synases
syn_exc_inh.w_ei = w_ei_

# Creating Synapse object for inh. -> exc. connection
syn_inh_exc = Synapses(neuron_group_inh, neuron_group_exc, model=syn_eqs_ie, on_pre=syn_on_pre_ie, method="euler")
syn_inh_exc.connect(i=syn_con_inh[1], j=syn_con_inh[0]) # inh. neurons connected to all exc. neurons expect the one which has the same index
print(f"------------------------------")
print(f"syn_inh_exc connection: ")
print(f"len-> {len(syn_inh_exc.i)}")
print(f"first 20 i -> {syn_inh_exc.i[:20]}")
print(f"first 20 j -> {syn_inh_exc.j[:20]}")

# Setting weight for inh. -> exc. synases
syn_inh_exc.w_ie = w_ie_

# Defining PoissonGroup for inputs
image_input = PoissonGroup(N=784, rates=0*Hz) # rates are changed according to image later


# Creating synapse object for input -> exc. connection, since inputs neurons are also excitatory we use 
# equations for exc. -> exc. (ee)
if test_phase:

    syn_input_exc = Synapses(image_input, neuron_group_exc, model=syn_eqs_ee_test, on_pre=syn_on_pre_ee_test, method="euler")
    syn_input_exc.connect(i=syn_con_exc[0], j=syn_con_exc[1])
    weights = np.load(f'{run_path}/input_to_exc_trained_weights.npy')
    syn_input_exc.w_ee[:] = weights # Setting pre-trained weights

else: # training phase
    syn_input_exc = Synapses(image_input, neuron_group_exc, model=syn_eqs_ee_training, on_pre=syn_on_pre_ee_training, on_post=syn_on_post_ee_training, method="euler")
    syn_input_exc.connect(i=syn_con_exc[0], j=syn_con_exc[1])
    print(f"------------------------------")
    print(f"syn_input_exc connection: ")
    print(f"len-> {len(syn_input_exc.i)}")
    print(f"first 20 i -> {syn_input_exc.i[:20]}")
    print(f"first 20 j -> {syn_input_exc.j[:20]}")

    syn_input_exc.w_ee[:] = "rand() * 0.3" # Initializing weights

syn_input_exc.delay = 10 * ms

# Defining SpikeMonitor to record spike counts of neuron in neuron_group_exc
spike_mon_ng_exc = SpikeMonitor(neuron_group_exc, record=True)

full_spike_mon_ng_exc = SpikeMonitor(neuron_group_exc, record=True)
poisson_spike_mon = SpikeMonitor(image_input, record=True)

syn_input_exc_mon = StateMonitor(syn_input_exc, ['w_ee'], record=True, dt=500*ms)

# Getting spiking rates and labels according to run_mode
image_input_rates, image_labels = get_spiking_rates_and_labels(test_phase, image_count, seed_data, max_rate)

run(0*ms)

curr_image_idx = 0                             # Tracks the index of the current image during iteration.

accuracies = []                                # This list is for saving calculated accuracies during training.

spike_counts_per_image = []                    # List to store the spike counts of each neuron for each image.
                                               # First dimension represents image idx and second dimension shows spike counts. 

max_rate_current_image = max_rate

for rc in range(run_count):
    while(curr_image_idx < image_count):  # While loop which will continue until all training data is finished.
        if curr_image_idx%50 == 0:
            print("----------------------------------")
            print(f"Current image: {curr_image_idx}")
            print(f"Elapsed time:", {time.time() - start})
            print("----------------------------------")
        image_input.rates = image_input_rates[curr_image_idx] * Hz   # Setting poisson neuron rates for current input image.

        #divisive_weight_normalization(syn_input_exc, population_exc) # Apply weight normalization

        run(350 * ms)  # training network for 350 ms.

        spike_counts_current_image = spike_mon_ng_exc.count[:]
        del spike_mon_ng_exc
        spike_mon_ng_exc = SpikeMonitor(neuron_group_exc, record=True)

        sum_spike_counts_current_image = sum(spike_counts_current_image) # TODO: naming convention needs checking

        # Calculate accuracy during training at determined intervals:
        if not sum_spike_counts_current_image < 5 and curr_image_idx % update_interval == 0 and curr_image_idx != 0:
            # Get image labels for current interval
            image_labels_curr_interval = image_labels[curr_image_idx - update_interval:curr_image_idx]
            if not test_phase:
                assign_neurons_to_labels(spike_counts_per_image, image_labels_curr_interval, population_exc, f"{run_path}")

            predictions_per_image = get_predictions(spike_counts_per_image, f"{run_path}")
            accuracy = calculate_accuracy(predictions_per_image, image_labels_curr_interval)

            accuracies.append(accuracy)
            
            draw_heatmap(full_spike_mon_ng_exc.count[:], f"{run_path}", f"R{run_count}_I{curr_image_idx}_exc1_spike")
            draw_heatmap(poisson_spike_mon.count[:], f"{run_path}", f"R{run_count}_I{curr_image_idx}_poisson_spike")

            draw_weights(syn_input_exc, population_exc, rf_size, f"{run_path}", f"R{run_count}_I{curr_image_idx}_syn_input_weights")
            # Reset spike_counts_per_image for new interval
            spike_counts_per_image = []

        if  sum_spike_counts_current_image < 5:
            # Input frequency for current image is increased by 32 Hz if sum of 
            # spike counts of all neurons for current image is smaller than 5 and
            # training is repeated again.
            max_rate_current_image += 32
            image_input_rates[curr_image_idx] = increase_spiking_rates(image_input_rates[curr_image_idx], max_rate_current_image)

            # Run simulation 150 ms without learning, before representing current image again.
            image_input.rates = 0 * Hz
            run(150 * ms)
        else:
            spike_counts_per_image.append(spike_counts_current_image)

            # Run simulation 150 ms without learning, before next image.
            image_input.rates = 0 * Hz
            run(150 * ms)

            # reset max_rate_current_image before presenting new image.
            max_rate_current_image = max_rate

            curr_image_idx += 1
    print("----------------------------------")
    print(f"{rc + 1}. iteration over dataset is finished.")
    # Calculate accuracy after iteration over dataset is finished.
    image_labels_curr_interval = image_labels[image_count - update_interval : image_count]

    if not test_phase:
        assign_neurons_to_labels(spike_counts_per_image, image_labels_curr_interval, population_exc, f"{run_path}")
    predictions_per_image = get_predictions(spike_counts_per_image, f"{run_path}")
    accuracy = calculate_accuracy(predictions_per_image, image_labels_curr_interval)

    accuracies.append(accuracy)

    # Reset curr_image_idx and spike_counts_per_image before giving dataset again.
    curr_image_idx = 0
    spike_counts_per_image = []

end = time.time()
sim_time = end - start
print(f"Simulation time: {sim_time}")

if not test_phase: # training phase
    # Save weights and theta values.
    weights = syn_input_exc.w_ee[:]
    np.save(f'{run_path}/input_to_exc_trained_weights.npy', weights)
    theta_values = neuron_group_exc.theta[:]
    np.save(f'{run_path}/theta_values_exc.npy', theta_values)

if test_phase:
    run_label = "test"
else:
    run_label = "training"
# iteration is x label of graph
iteration = [rc * image_count + img_idx for rc in range(run_count) for img_idx in range(update_interval, image_count+1, update_interval)]

plt.figure(100)
plt.plot(iteration, accuracies)
plt.title(f'Accuracy change over iterations for {run_label} phase')
plt.xlabel("Iteration Count")
plt.ylabel("Accuracy % ")
plt.grid(True)
plt.savefig(f'{run_path}/{run_label}_accuracy_graph.png')

draw_heatmap(full_spike_mon_ng_exc.count[:], f"{run_path}", "final_exc1_spikes")
draw_heatmap(poisson_spike_mon.count[:], f"{run_path}", "final_poisson_spikes")

draw_weights(syn_input_exc, population_exc, rf_size, f"{run_path}", f"final_syn_input_weights")
write_to_csv(args, accuracies[-1], run_name, sim_time)
