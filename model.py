from brian2 import *
from equations import Equations
from util.parser_util import get_args
from util.dump_util import load_data

class Model:

    def __init__(self):
        self.eqs = Equations()
        self.args = get_args()
        self.mode = "test" if self.args.test_phase else "train"
        self.run_path = f"runs/{self.args.run_name}"
        self.spike_mon_dump_path = f"{self.run_path}/spike_mon_dump"
        self.weight_dump_path = f"{self.run_path}/weight_dump"
        self.theta_dump_path = f"{self.run_path}/theta_dump"
        self.acc_dump_path = f"{self.run_path}/acc_dump"
        self.model_dump_path = f"{self.run_path}/model_dump"
        self.layout = int(sqrt(self.args.population_exc))

        self.ng_eqs_exc = self.eqs.ng_eqs_exc
        self.ng_eqs_inh = self.eqs.ng_eqs_inh
        self.ng_threshold_exc = self.eqs.ng_threshold_exc
        self.ng_threshold_inh = self.eqs.ng_threshold_inh
        self.ng_reset_exc = self.eqs.ng_reset_exc
        self.ng_reset_inh = self.eqs.ng_reset_inh
        self.ee_syn_eqs = ""
        self.ee_syn_on_pre = ""
        self.ee_syn_on_post = ""
        self.ei_syn_eqs = self.eqs.syn_eqs_ei
        self.ei_syn_on_pre = self.eqs.syn_on_pre_ei
        self.ie_syn_eqs = self.eqs.syn_eqs_ie
        self.ie_syn_on_pre = self.eqs.syn_on_pre_ie

        self.update_equations()

    def update_equations(self):
        if self.args.test_phase:
            self.ng_eqs_exc += "theta : volt"
            self.ee_syn_eqs = self.eqs.syn_eqs_ee_test
            self.ee_syn_on_pre = self.eqs.syn_on_pre_ee_test
        else:
            self.ng_eqs_exc += "dtheta/dt = -theta/tau_theta  : volt"
            self.ng_reset_exc += "theta += theta_inc_exc"
            self.ee_syn_eqs = self.eqs.syn_eqs_ee_training
            self.ee_syn_on_pre = self.eqs.syn_on_pre_ee_training
            self.ee_syn_on_post= self.eqs.syn_on_post_ee_training

    def exc_ng_initial_vals(self, ng_idx, ng_exc):
        if self.args.test_phase:
            theta_values_exc = load_data(f"{self.theta_dump_path}/theta_ng{ng_idx}_train")
            ng_exc.theta = theta_values_exc * volt
        else:
            ng_exc.theta = 20 * mV

        ng_exc.v = self.args.E_rest_exc * mV - 40 * mV

    def inh_ng_initial_vals(self, ng_inh):
        ng_inh.v = self.args.E_rest_inh * mV - 40 * mV

    def ei_syn_initial_vals(self, ei_syn):
        ei_syn.w_ei = self.args.w_ei_

    def ie_syn_initial_vals(self, ie_syn):
        ie_syn.w_ie = self.args.w_ie_

    def ee_syn_initial_vals(self, syn_idx, ee_syn):
        if self.args.test_phase:
            weights = load_data(f"{self.weight_dump_path}/ee_syn{syn_idx}_train/final_ee_syn{syn_idx}_train")
            ee_syn.w_ee[:] = weights  # Setting pre-trained weights
        else:
            ee_syn.w_ee[:] = "rand() * 0.3"  # Initializing weights
        ee_syn.delay = self.args.delay_ee * ms

    # Set NeuronGroup Parameters:
    def set_ng_namespace(self, ng: NeuronGroup):
        ng.namespace["E_rest_exc"] = self.args.E_rest_exc * mV
        ng.namespace["E_rest_inh"] = self.args.E_rest_inh * mV
        ng.namespace["E_exc_for_exc"] = self.args.E_exc_for_exc * mV
        ng.namespace["E_inh_for_exc"] = self.args.E_inh_for_exc * mV
        ng.namespace["E_exc_for_inh"] = self.args.E_exc_for_inh * mV
        ng.namespace["E_inh_for_inh"] = self.args.E_inh_for_inh * mV
        ng.namespace["tau_lif_exc"] = self.args.tau_lif_exc * ms
        ng.namespace["tau_lif_inh"] = self.args.tau_lif_inh * ms
        ng.namespace["tau_ge"] = self.args.tau_ge * ms
        ng.namespace["tau_gi"] = self.args.tau_gi * ms
        ng.namespace["tau_theta"] = self.args.tau_theta * ms
        ng.namespace["theta_inc_exc"] = self.args.theta_inc_exc * mV
        ng.namespace["v_threshold_exc"] = self.args.v_threshold_exc * mV
        ng.namespace["v_threshold_inh"] = self.args.v_threshold_inh * mV
        ng.namespace["v_offset_exc"] = self.args.v_offset_exc * mV
        ng.namespace["v_reset_exc"] = self.args.v_reset_exc * mV
        ng.namespace["v_reset_inh"] = self.args.v_reset_inh * mV

    # Set Synapse Parameters:
    def set_syn_namespace(self, syn: Synapses):
        syn.namespace["tau_Apre_ee"] = self.args.tau_Apre_ee * ms
        syn.namespace["tau_Apost1_ee"] = self.args.tau_Apost1_ee * ms
        syn.namespace["tau_Apost2_ee"] = self.args.tau_Apost2_ee * ms
        syn.namespace["eta_pre_ee"] = self.args.eta_pre_ee
        syn.namespace["eta_post_ee"] = self.args.eta_post_ee
        syn.namespace["w_min_ee"] = self.args.w_min_ee
        syn.namespace["w_max_ee"] = self.args.w_max_ee
        syn.namespace["w_ei_"] = self.args.w_ei_
        syn.namespace["w_ie_"] = self.args.w_ie_
        syn.namespace["g_e_multiplier"] = self.args.g_e_multiplier