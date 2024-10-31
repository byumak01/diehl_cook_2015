from dataclasses import dataclass


@dataclass
class Equations:
    # NeuronGroup equations for exc. and inh. populations
    ng_eqs_exc = """
    dv/dt = ((E_rest_exc - v) + g_e*(E_exc_for_exc - v) + g_i*(E_inh_for_exc - v))/tau_lif_exc : volt (unless refractory)  # (Eq. 1 in the paper)
    dg_e/dt = -g_e/tau_ge : 1                                                                                              # (Eq. 2 in the paper)
    dg_i/dt = -g_i/tau_gi : 1                                                                                              # (Eq. 2 in the paper (Inhibitory version))
    """

    # Theta is used for adaptive threshold mechanism. It uses its trained value in test phase.
    # w_sum is used in divisive weight normalization.

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
    g_e_post += w_ee * g_e_multiplier
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