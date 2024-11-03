import argparse
from datetime import datetime


def str_to_float_list(val):
    return [float(item) for item in val.split(',')]


def str_to_int_list(val):
    return [int(item) for item in val.split(',')]


def check_args(args):
    args_dict = vars(args)
    for param_name, val in args_dict.items():
        if isinstance(val, list) and len(val) != args.layer_count and len(val) != 1:
            raise ValueError(f"{param_name} should have 1 or {args.layer_count} elements, but got {len(val)}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Neuron, Synapse, and PoissonGroup parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add NeuronGroup parameters
    parser.add_argument('--E_rest_exc', type=str_to_float_list, default=[-65],
                        help="Resting potential for excitatory neurons (mV)")
    parser.add_argument('--E_rest_inh', type=str_to_float_list, default=[-60],
                        help="Resting potential for inhibitory neurons (mV)")
    parser.add_argument('--E_exc_for_exc', type=str_to_float_list, default=[0],
                        help="Excitatory reversal potential (mV)")
    parser.add_argument('--E_inh_for_exc', type=str_to_float_list, default=[-100],
                        help="Inhibitory reversal potential (mV)")
    parser.add_argument('--E_exc_for_inh', type=str_to_float_list, default=[0],
                        help="Excitatory reversal potential for inhibitory neurons (mV)")
    parser.add_argument('--E_inh_for_inh', type=str_to_float_list, default=[-85],
                        help="Inhibitory reversal potential for inhibitory neurons (mV)")
    parser.add_argument('--tau_lif_exc', type=str_to_float_list, default=[100],
                        help="LIF decay for excitatory neurons (ms)")
    parser.add_argument('--tau_lif_inh', type=str_to_float_list, default=[10],
                        help="LIF decay for inhibitory neurons (ms)")
    parser.add_argument('--tau_ge', type=str_to_float_list, default=[1], help="Excitatory conductance decay (ms)")
    parser.add_argument('--tau_gi', type=str_to_float_list, default=[2], help="Inhibitory conductance decay (ms)")
    parser.add_argument('--tau_theta', type=str_to_float_list, default=[1e7], help="Theta decay rate (ms)")
    parser.add_argument('--theta_inc_exc', type=str_to_float_list, default=[0.05],
                        help="Theta increment for excitatory neurons (mV)")
    parser.add_argument('--refractory_exc', type=str_to_float_list, default=[5],
                        help="Refractory period for excitatory neurons (ms)")
    parser.add_argument('--refractory_inh', type=str_to_float_list, default=[2],
                        help="Refractory period for inhibitory neurons (ms)")
    parser.add_argument('--v_threshold_exc', type=str_to_float_list, default=[-52],
                        help="Spiking threshold for excitatory neurons (mV)")
    parser.add_argument('--v_threshold_inh', type=str_to_float_list, default=[-40],
                        help="Spiking threshold for inhibitory neurons (mV)")
    parser.add_argument('--v_offset_exc', type=str_to_float_list, default=[20],
                        help="Threshold offset for excitatory neurons (mV)")
    parser.add_argument('--v_reset_exc', type=str_to_float_list, default=[-65],
                        help="Reset voltage for excitatory neurons (mV)")
    parser.add_argument('--v_reset_inh', type=str_to_float_list, default=[-45],
                        help="Reset voltage for inhibitory neurons (mV)")
    parser.add_argument('--population_exc', type=str_to_int_list, default=[784],
                        help="Population of excitatory neurons")
    parser.add_argument('--population_inh', type=str_to_int_list, default=[784],
                        help="Population of inhibitory neurons")

    # Synapse parameters
    parser.add_argument('--tau_Apre_ee', type=str_to_float_list, default=[20],
                        help="Apre decay for exc.->exc. synapse (ms)")
    parser.add_argument('--tau_Apost1_ee', type=str_to_float_list, default=[20],
                        help="Apost1 decay for exc.->exc. synapse (ms)")
    parser.add_argument('--tau_Apost2_ee', type=str_to_float_list, default=[40],
                        help="Apost2 decay for exc.->exc. synapse (ms)")
    parser.add_argument('--eta_pre_ee', type=str_to_float_list, default=[0.0001],
                        help="Pre-synaptic learning rate for exc.->exc. synapse")
    parser.add_argument('--eta_post_ee', type=str_to_float_list, default=[0.01],
                        help="Post-synaptic learning rate for exc.->exc. synapse")
    parser.add_argument('--w_min_ee', type=str_to_float_list, default=[0], help="Minimum weight for exc.->exc. synapse")
    parser.add_argument('--w_max_ee', type=str_to_float_list, default=[1], help="Maximum weight for exc.->exc. synapse")
    parser.add_argument('--w_ei_', type=str_to_float_list, default=[10.4], help="Weight for exc.->inh. synapse")
    parser.add_argument('--w_ie_', type=str_to_float_list, default=[17], help="Weight for inh.->exc. synapse")
    parser.add_argument('--delay_ee', type=str_to_float_list, default=[10], help="Delay for exc.->exc. synapse (ms)")
    parser.add_argument('--g_e_multiplier', type=str_to_float_list, default=[1],
                        help="g_e_multiplier (on_pre -> g_e_post = w_ee * g_e_multiplier)")
    parser.add_argument('--normalization_const', type=str_to_float_list, default=[78],
                        help="Normalization constant for div. w. norm.")
    # Add PoissonGroup parameters
    parser.add_argument('--max_rate', type=float, default=63.75, help="Maximum rate for PoissonGroup (Hz)")
    # Other params
    parser.add_argument('--seed_data', action='store_true', help="Set this flag to seed the data")
    parser.add_argument('--rf_size', type=str_to_int_list, default=[5], help="RF size of neurons")
    parser.add_argument('--test_phase', action='store_true', help="Set this flag to indicate test_phase")
    parser.add_argument('--run_count', type=int, default=1, help="How many times dataset will be iterated")
    parser.add_argument('--layer_count', type=int, default=1, help="How many NG layers there should be")
    parser.add_argument('--image_count', type=int, default=5000, help="How many images will be used for run")
    parser.add_argument('--draw_update_interval', type=int, default=500, help="Update interval for heatmaps")
    parser.add_argument('--acc_update_interval', type=int, default=500, help="Update interval for accuracy")
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%m%d_%H%M%S"), help="run name")

    return parser.parse_args()
