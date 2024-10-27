from brian2 import *

def create_neuron_groups(model):
    exc_neuron_groups = []
    inh_neuron_groups = []
    for i in range(model.args.layer_count):
        neuron_group_exc = NeuronGroup(N=model.args.population_exc, model=model.ng_eqs_exc,
                                       threshold=model.ng_threshold_exc, reset=model.ng_reset_exc,
                                       refractory=model.args.refractory_exc * ms, method="euler")
        neuron_group_inh = NeuronGroup(N=model.args.population_inh, model=model.ng_eqs_inh,
                                       threshold=model.ng_threshold_inh, reset=model.ng_reset_inh,
                                       refractory=model.args.refractory_inh * ms, method="euler")

        # Setting initial values for exc. and inh. populations
        model.set_ng_namespace(neuron_group_exc)
        model.set_ng_namespace(neuron_group_inh)

        model.exc_ng_initial_vals(i, neuron_group_exc)
        model.inh_ng_initial_vals(neuron_group_inh)
        exc_neuron_groups.append(neuron_group_exc)
        inh_neuron_groups.append(neuron_group_inh)

    return exc_neuron_groups, inh_neuron_groups