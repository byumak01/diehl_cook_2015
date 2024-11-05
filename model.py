from brian2 import *
from equations import Equations
from util.parser_util import get_args, check_args, get_param
from util.dump_util import load_data
import logging

model_logger = logging.getLogger('base.model')


class Model:

    def __init__(self):
        self.eqs = Equations()
        self.args = get_args()
        check_args(self.args)
        self.mode = "test" if self.args.test_phase else "train"
        self.run_path = f"runs/{self.args.run_name}"
        self.spike_mon_dump_path = f"{self.run_path}/spike_mon_dump"
        self.weight_dump_path = f"{self.run_path}/weight_dump"
        self.theta_dump_path = f"{self.run_path}/theta_dump"
        self.acc_dump_path = f"{self.run_path}/acc_dump"
        self.model_dump_path = f"{self.run_path}/model_dump"

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
        model_logger.debug(f"\n ng_eqs_exc: {self.ng_eqs_exc}")
        model_logger.debug(f"\n ng_eqs_inh: {self.ng_threshold_exc}")
        model_logger.debug(f"\n ng_threshold_exc: {self.ng_threshold_exc}")
        model_logger.debug(f"\n ng_threshold_inh: {self.ng_threshold_inh}")
        model_logger.debug(f"\n ng_reset_exc: {self.ng_reset_exc}")
        model_logger.debug(f"\n ng_reset_inh: {self.ng_reset_inh}")
        model_logger.debug(f"\n ee_syn_eqs: {self.ee_syn_eqs}")
        model_logger.debug(f"\n ee_syn_on_pre: {self.ee_syn_on_pre}")
        model_logger.debug(f"\n ee_syn_on_post: {self.ee_syn_on_post}")
        model_logger.debug(f"\n ie_syn_eqs: {self.ie_syn_eqs}")
        model_logger.debug(f"\n ie_syn_on_pre: {self.ie_syn_on_pre}")
        model_logger.debug(f"\n ei_syn_eqs: {self.ei_syn_eqs}")
        model_logger.debug(f"\n ei_syn_on_pre: {self.ei_syn_on_pre}")

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
            self.ee_syn_on_post = self.eqs.syn_on_post_ee_training

    def exc_ng_initial_vals(self, ng_idx, ng_exc):
        if self.args.test_phase:
            theta_values_exc = load_data(f"{self.theta_dump_path}/final_theta_ng{ng_idx}_train")
            ng_exc.theta = theta_values_exc
        else:
            ng_exc.theta = get_param(self.args.theta, ng_idx) * mV

        model_logger.debug(f"ng_exc.theta: {ng_exc.theta[0:10]}")

        ng_exc.v = get_param(self.args.E_rest_exc, ng_idx) * mV - 40 * mV
        model_logger.debug(f"ng_exc.v: {ng_exc.v[0]}")

    def inh_ng_initial_vals(self, idx: int, ng_inh):
        ng_inh.v = get_param(self.args.E_rest_inh, idx) * mV - 40 * mV
        model_logger.debug(f"ng_inh.v: {ng_inh.v[0]}")

    def ei_syn_initial_vals(self, idx: int, ei_syn):
        ei_syn.w_ei = get_param(self.args.w_ei_, idx)
        model_logger.debug(f"ei_syn.w_ei: {ei_syn.w_ei[0]}")

    def ie_syn_initial_vals(self, idx: int, ie_syn):
        ie_syn.w_ie = get_param(self.args.w_ie_, idx)
        model_logger.debug(f"ie_syn.w_ie: {ie_syn.w_ie[0]}")

    def ee_syn_initial_vals(self, syn_idx, ee_syn):
        if self.args.test_phase:
            weights = load_data(f"{self.weight_dump_path}/ee_syn{syn_idx}_train/final_ee_syn{syn_idx}_train")
            ee_syn.w_ee[:] = weights['w_ee']  # Setting pre-trained weights
        else:
            ee_syn.w_ee[:] = "rand() * 0.3"  # Initializing weights
        model_logger.debug(f"ee_syn.w_ee: {ee_syn.w_ee[:10]}")
        ee_syn.delay = get_param(self.args.delay_ee, syn_idx) * ms
        model_logger.debug(f"ee_syn.delay: {ee_syn.delay[0]}")

    # Set NeuronGroup Parameters:
    def set_ng_namespace(self, idx: int, ng: NeuronGroup):
        model_logger.debug(f"ng_namespace, idx: {idx}")
        ng.namespace["E_rest_exc"] = get_param(self.args.E_rest_exc, idx) * mV
        model_logger.debug(f"ng.E_rest_exc: {ng.namespace['E_rest_exc']}")
        ng.namespace["E_rest_inh"] = get_param(self.args.E_rest_inh, idx) * mV
        model_logger.debug(f"ng.E_rest_inh: {ng.namespace['E_rest_inh']}")
        ng.namespace["E_exc_for_exc"] = get_param(self.args.E_exc_for_exc, idx) * mV
        model_logger.debug(f"ng.E_exc_for_exc: {ng.namespace['E_exc_for_exc']}")
        ng.namespace["E_inh_for_exc"] = get_param(self.args.E_inh_for_exc, idx) * mV
        model_logger.debug(f"ng.E_inh_for_exc: {ng.namespace['E_inh_for_exc']}")
        ng.namespace["E_exc_for_inh"] = get_param(self.args.E_exc_for_inh, idx) * mV
        model_logger.debug(f"ng.E_exc_for_inh: {ng.namespace['E_exc_for_inh']}")
        ng.namespace["E_inh_for_inh"] = get_param(self.args.E_inh_for_inh, idx) * mV
        model_logger.debug(f"ng.E_inh_for_inh: {ng.namespace['E_inh_for_inh']}")
        ng.namespace["tau_lif_exc"] = get_param(self.args.tau_lif_exc, idx) * ms
        model_logger.debug(f"tau_lif_exc: {ng.namespace['tau_lif_exc']}")
        ng.namespace["tau_lif_inh"] = get_param(self.args.tau_lif_inh, idx) * ms
        model_logger.debug(f"tau_lif_inh: {ng.namespace['tau_lif_inh']}")
        ng.namespace["tau_ge"] = get_param(self.args.tau_ge, idx) * ms
        model_logger.debug(f"tau_ge: {ng.namespace['tau_ge']}")
        ng.namespace["tau_gi"] = get_param(self.args.tau_gi, idx) * ms
        model_logger.debug(f"tau_gi: {ng.namespace['tau_gi']}")
        ng.namespace["tau_theta"] = get_param(self.args.tau_theta, idx) * ms
        model_logger.debug(f"tau_theta: {ng.namespace['tau_theta']}")
        ng.namespace["theta_inc_exc"] = get_param(self.args.theta_inc_exc, idx) * mV
        model_logger.debug(f"theta_inc_exc: {ng.namespace['theta_inc_exc']}")
        ng.namespace["v_threshold_exc"] = get_param(self.args.v_threshold_exc, idx) * mV
        model_logger.debug(f"v_threshold_exc: {ng.namespace['v_threshold_exc']}")
        ng.namespace["v_threshold_inh"] = get_param(self.args.v_threshold_inh, idx) * mV
        model_logger.debug(f"v_threshold_inh: {ng.namespace['v_threshold_inh']}")
        ng.namespace["v_offset_exc"] = get_param(self.args.v_offset_exc, idx) * mV
        model_logger.debug(f"v_offset_exc: {ng.namespace['v_offset_exc']}")
        ng.namespace["v_reset_exc"] = get_param(self.args.v_reset_exc, idx) * mV
        model_logger.debug(f"v_reset_exc: {ng.namespace['v_reset_exc']}")
        ng.namespace["v_reset_inh"] = get_param(self.args.v_reset_inh, idx) * mV
        model_logger.debug(f"v_reset_inh: {ng.namespace['v_reset_inh']}")

    # Set Synapse Parameters:
    def set_syn_namespace(self, idx: int, syn: Synapses):
        model_logger.debug(f"syn_namespace, idx: {idx}")
        syn.namespace["tau_Apre_ee"] = get_param(self.args.tau_Apre_ee, idx) * ms
        model_logger.debug(f"tau_Apre_ee: {syn.namespace['tau_Apre_ee']}")
        syn.namespace["tau_Apost1_ee"] = get_param(self.args.tau_Apost1_ee, idx) * ms
        model_logger.debug(f"tau_Apost1_ee: {syn.namespace['tau_Apost1_ee']}")
        syn.namespace["tau_Apost2_ee"] = get_param(self.args.tau_Apost2_ee, idx) * ms
        model_logger.debug(f"tau_Apost2_ee: {syn.namespace['tau_Apost2_ee']}")
        syn.namespace["eta_pre_ee"] = get_param(self.args.eta_pre_ee, idx)
        model_logger.debug(f"eta_pre_ee: {syn.namespace['eta_pre_ee']}")
        syn.namespace["eta_post_ee"] = get_param(self.args.eta_post_ee, idx)
        model_logger.debug(f"eta_post_ee: {syn.namespace['eta_post_ee']}")
        syn.namespace["w_min_ee"] = get_param(self.args.w_min_ee, idx)
        model_logger.debug(f"w_min_ee: {syn.namespace['w_min_ee']}")
        syn.namespace["w_max_ee"] = get_param(self.args.w_max_ee, idx)
        model_logger.debug(f"w_max_ee: {syn.namespace['w_max_ee']}")
        syn.namespace["w_ei_"] = get_param(self.args.w_ei_, idx)
        model_logger.debug(f"w_ei_: {syn.namespace['w_ei_']}")
        syn.namespace["w_ie_"] = get_param(self.args.w_ie_, idx)
        model_logger.debug(f"w_ie_: {syn.namespace['w_ie_']}")
        syn.namespace["g_e_multiplier"] = get_param(self.args.g_e_multiplier, idx)
        model_logger.debug(f"g_e_multiplier: {syn.namespace['g_e_multiplier']}")

