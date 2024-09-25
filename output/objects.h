
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>


namespace brian {

extern std::string results_dir;
// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;



void set_variable_by_name(std::string, std::string);

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_1_delay;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_1_w_ie;
extern std::vector<int32_t> _dynamic_array_synapses_2__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_2__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_2_Apost1_ee;
extern std::vector<double> _dynamic_array_synapses_2_Apost2_ee;
extern std::vector<double> _dynamic_array_synapses_2_Apost2_prev_ee;
extern std::vector<double> _dynamic_array_synapses_2_Apre_ee;
extern std::vector<double> _dynamic_array_synapses_2_delay;
extern std::vector<double> _dynamic_array_synapses_2_delay_1;
extern std::vector<double> _dynamic_array_synapses_2_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_2_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_2_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_2_w_ee;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_delay;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_w_ei;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup_1__spikespace;
extern const int _num__array_neurongroup_1__spikespace;
extern double *_array_neurongroup_1_g_e;
extern const int _num__array_neurongroup_1_g_e;
extern double *_array_neurongroup_1_g_i;
extern const int _num__array_neurongroup_1_g_i;
extern int32_t *_array_neurongroup_1_i;
extern const int _num__array_neurongroup_1_i;
extern double *_array_neurongroup_1_lastspike;
extern const int _num__array_neurongroup_1_lastspike;
extern char *_array_neurongroup_1_not_refractory;
extern const int _num__array_neurongroup_1_not_refractory;
extern double *_array_neurongroup_1_v;
extern const int _num__array_neurongroup_1_v;
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern double *_array_neurongroup_g_e;
extern const int _num__array_neurongroup_g_e;
extern double *_array_neurongroup_g_i;
extern const int _num__array_neurongroup_g_i;
extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double *_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;
extern char *_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;
extern double *_array_neurongroup_theta;
extern const int _num__array_neurongroup_theta;
extern double *_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern int32_t *_array_poissongroup__spikespace;
extern const int _num__array_poissongroup__spikespace;
extern int32_t *_array_poissongroup_i;
extern const int _num__array_poissongroup_i;
extern int32_t *_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t *_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t *_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;
extern int32_t *_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t *_array_synapses_2_N;
extern const int _num__array_synapses_2_N;
extern int32_t *_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
extern double *_timedarray_values;
extern const int _num__timedarray_values;

//////////////// synapses /////////////////
// synapses
extern SynapticPathway synapses_pre;
// synapses_1
extern SynapticPathway synapses_1_pre;
// synapses_2
extern SynapticPathway synapses_2_post;
extern SynapticPathway synapses_2_pre;

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


