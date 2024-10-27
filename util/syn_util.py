from brian2 import *


def normalize_synapses(model, synapses: list[Synapses]):
    for syn in synapses:
        _divisive_weight_normalization(model, syn)


def _divisive_weight_normalization(model, synapse: Synapses) -> None:
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


def receptive_field_for_exc(model, neuron_idx: int):
    half_size = model.args.rf_size // 2
    return [model.layout * j + i for j in
            range((neuron_idx // model.layout) - half_size, (neuron_idx // model.layout) + half_size + 1) for
            i in range((neuron_idx % model.layout) - half_size, (neuron_idx % model.layout) + half_size + 1)
            if 0 <= i < model.layout and 0 <= j < model.layout]


def receptive_field_for_inh(model, neuron_idx: int):
    half_size = model.args.rf_size // 2
    return [model.layout * j + i for j in
            range((neuron_idx // model.layout) - half_size, (neuron_idx // model.layout) + half_size + 1) for
            i in range((neuron_idx % model.layout) - half_size, (neuron_idx % model.layout) + half_size + 1)
            if 0 <= i < model.layout and 0 <= j < model.layout and model.layout * j + i != neuron_idx]


def synapse_connections_exc(model):
    return np.transpose([(x, i) for i in range(model.args.population_exc) for x in receptive_field_for_exc(model, i)])


def synapse_connections_inh(model):
    return np.transpose([(x, i) for i in range(model.args.population_inh) for x in receptive_field_for_inh(model, i)])


def package_syn_data(syn: Synapses):
    package = {"w_ee": syn.w_ee[:], "i": syn.i[:], "j": syn.j[:]}
    return package


def create_synapses_ei(model, exc_neuron_groups, inh_neuron_groups):
    ei_synapses = []
    for exc_neuron_group, inh_neuron_group in zip(exc_neuron_groups, inh_neuron_groups):
        syn_exc_inh = Synapses(exc_neuron_group, inh_neuron_group, model=model.ei_syn_eqs, on_pre=model.ei_syn_on_pre,
                               method="euler")
        syn_exc_inh.connect(j='i')  # One-to-one connection

        model.set_syn_namespace(syn_exc_inh)
        model.ei_syn_initial_vals(syn_exc_inh)
        ei_synapses.append(syn_exc_inh)
    return ei_synapses


def create_synapses_ie(model, exc_neuron_groups, inh_neuron_groups):
    syn_con_inh = synapse_connections_inh(model)
    ie_synapses = []
    for exc_neuron_group, inh_neuron_group in zip(exc_neuron_groups, inh_neuron_groups):
        syn_inh_exc = Synapses(inh_neuron_group, exc_neuron_group, model=model.ie_syn_eqs, on_pre=model.ie_syn_on_pre,
                               method="euler")
        syn_inh_exc.connect(i=syn_con_inh[1], j=syn_con_inh[0])

        model.set_syn_namespace(syn_inh_exc)
        model.ie_syn_initial_vals(syn_inh_exc)
        ie_synapses.append(syn_inh_exc)
    return ie_synapses


def create_synapses_ee(model, image_input, exc_neuron_groups):
    syn_con_exc = synapse_connections_exc(model)
    ee_synapses = []
    for group_idx in range(model.args.layer_count):
        source = exc_neuron_groups[group_idx - 1] if group_idx != 0 else image_input
        target = exc_neuron_groups[group_idx] if group_idx != 0 else exc_neuron_groups[0]
        syn_ee = Synapses(source, target, model=model.ee_syn_eqs, on_pre=model.ee_syn_on_pre,
                          on_post=model.ee_syn_on_post, method="euler")
        syn_ee.connect(i=syn_con_exc[0], j=syn_con_exc[1])

        model.set_syn_namespace(syn_ee)
        model.ee_syn_initial_vals(group_idx, syn_ee)
        ee_synapses.append(syn_ee)
    return ee_synapses
