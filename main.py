"""
This script is a reproduction of the work presented in:

Original Research Article:
Peter U. Diehl, Matthew Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity," 
Frontiers in Computational Neuroscience, vol. 9, 2015. 
DOI: https://doi.org/10.3389/fncom.2015.00099

Original code available at: https://github.com/peter-u-diehl/stdp-mnist/tree/master

Rewritten by: Barış Yumak, 2024

python main.py --seed_data --image_count 2500 --acc_update_interval 500 --draw_update_interval 500 --g_e_multiplier 3 --normalization_const 9 --rf_size 9
"""

from util.dump_util import ensure_path, dump_data, dump_theta_values, write_to_csv, dump_weights
from util.dataset_util import get_spiking_rates_and_labels
from util.ng_util import create_neuron_groups
from util.sim_util import check_update
from util.poisson_util import increase_spiking_rates
from util.syn_util import create_synapses_ee, create_synapses_ie, create_synapses_ei, package_syn_data, \
    normalize_synapses
from evaluation import get_accuracy, acc_update
from model import Model
from brian2 import *

model = Model()

if model.args.test_phase and not os.path.exists(model.run_path):
    raise ValueError(f"There isn't a run named {model.args.run_name} or folder is empty. Cannot run test phase.")

if not model.args.test_phase:
    ensure_path(model.run_path)
    print(f"Directory {model.run_path} created successfully.")

start = time.time()

# Creating NeuronGroup objects for exc. and inh. populations
exc_neuron_groups, inh_neuron_groups = create_neuron_groups(model)

# Creating Synapse object for exc. -> inh. connection
ei_synapses = create_synapses_ei(model, exc_neuron_groups, inh_neuron_groups)

# Creating Synapse object for inh. -> exc. connection
ie_synapses = create_synapses_ie(model, exc_neuron_groups, inh_neuron_groups)
# Defining PoissonGroup for inputs
image_input = PoissonGroup(N=784, rates=0 * Hz)  # rates are changed according to image later

# Creating synapse object for input -> exc. connection, since inputs neurons are also excitatory we use
# equations for exc. -> exc. (ee)
ee_synapses = create_synapses_ee(model, image_input, exc_neuron_groups)

# Defining SpikeMonitor to record spike counts of neuron in neuron_group_exc
spk_mon_last_layer = SpikeMonitor(exc_neuron_groups[-1], record=True)
spk_mon_inh= SpikeMonitor(inh_neuron_groups[-1], record=True)
spk_mon_last_layer_dump_path = f"{model.spike_mon_dump_path}/{model.mode}/last_layer"

spk_mon_input = SpikeMonitor(image_input, record=True)
spk_mon_input_dump_path = f"{model.spike_mon_dump_path}/{model.mode}/pg"
# state_mon_syn_input_exc = StateMonitor(syn_input_exc, ['w_ee'], record=True, dt=2500 * 500 * ms)

# Getting spiking rates and labels according to run_mode
image_input_rates, image_labels = get_spiking_rates_and_labels(model)

net = Network(collect())
net.add(ee_synapses, ei_synapses, ie_synapses, exc_neuron_groups, inh_neuron_groups)
net.run(0 * ms)

curr_image_idx = 0  # Tracks the index of the current image during iteration.

accuracies = []  # This list is for saving calculated accuracies during training.

temp_spike_counts = 0

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
        # Setting poisson neuron rates for current input image.
        image_input.rates = image_input_rates[curr_image_idx] * Hz

        normalize_synapses(model, ee_synapses)  # Apply weight normalization

        net.run(350 * ms)  # training network for 350 ms.

        spike_counts_current_image = np.copy(spk_mon_last_layer.count[:]) - temp_spike_counts
        temp_spike_counts = np.copy(spk_mon_last_layer.count[:])


        sum_spike_counts_current_image = sum(spike_counts_current_image)

        if sum_spike_counts_current_image < 5:
            max_rate_current_image += 32
            image_input_rates[curr_image_idx] = increase_spiking_rates(image_input_rates[curr_image_idx],
                                                                       max_rate_current_image)

        else:
            is_acc_update = check_update(curr_image_idx, model.args.acc_update_interval)
            if is_acc_update:
                spike_counts_per_image = acc_update(model, curr_image_idx, image_labels, spike_counts_per_image,
                                                    accuracies)

            is_dump_draw_data = check_update(curr_image_idx, model.args.draw_update_interval)
            if is_dump_draw_data:
                dump_data(spk_mon_last_layer.count[:], spk_mon_last_layer_dump_path, f"R{rc}_I{curr_image_idx}_exc1")
                dump_data(spk_mon_input.count[:], spk_mon_input_dump_path, f"R{rc}_I{curr_image_idx}_pg")
                if not model.args.test_phase:
                    dump_theta_values(model, exc_neuron_groups, f"R{rc}_I{curr_image_idx}")
                    dump_weights(model, ee_synapses, f"R{rc}_I{curr_image_idx}")

            spike_counts_per_image.append(spike_counts_current_image)

            max_rate_current_image = model.args.max_rate

            curr_image_idx += 1

        # Run simulation 150 ms without learning, before next image.
        image_input.rates = 0 * Hz
        net.run(150 * ms)

    print("----------------------------------")
    print(f"{rc + 1}. iteration over dataset is finished.")
    # Calculate accuracy after iteration over dataset is finished.
    image_labels_curr_interval = image_labels[
                                 model.args.image_count - model.args.acc_update_interval: model.args.image_count]

    accuracy = get_accuracy(model, spike_counts_per_image, image_labels_curr_interval)
    accuracies.append(accuracy)

    # Reset curr_image_idx and spike_counts_per_image before giving dataset again.
    curr_image_idx = 0
    spike_counts_per_image = []

end = time.time()
sim_time = end - start
print(f"Simulation time: {sim_time}")

if not model.args.test_phase:  # training phase
    dump_theta_values(model, exc_neuron_groups, "final")
    dump_weights(model, ee_synapses, "final")

dump_data(accuracies, f"{model.acc_dump_path}", f"accuracies_{model.mode}")
dump_data(spk_mon_last_layer.count[:], spk_mon_last_layer_dump_path, f"final_exc1")
dump_data(spk_mon_input.count[:], spk_mon_input_dump_path, f"final_pg")
dump_data(model, f"{model.model_dump_path}", f"model_{model.mode}")
write_to_csv(model, accuracies[-1], sim_time)
