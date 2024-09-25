

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg
std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
Network magicnetwork;

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void set_variable_by_name(std::string name, std::string s_value) {
	size_t var_size;
	size_t data_size;
	// C-style or Python-style capitalization is allowed for boolean values
    if (s_value == "true" || s_value == "True")
        s_value = "1";
    else if (s_value == "false" || s_value == "False")
        s_value = "0";
	// non-dynamic arrays
    if (name == "neurongroup_1._spikespace") {
        var_size = 401;
        data_size = 401*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup_1__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.g_e") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_g_e, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_g_e, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.g_i") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_g_i, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_g_i, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.lastspike") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.not_refractory") {
        var_size = 400;
        data_size = 400*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_neurongroup_1_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.v") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_v, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup._spikespace") {
        var_size = 401;
        data_size = 401*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.g_e") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_g_e, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_g_e, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.g_i") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_g_i, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_g_i, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.lastspike") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.not_refractory") {
        var_size = 400;
        data_size = 400*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_neurongroup_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.theta") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_theta, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_theta, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.v") {
        var_size = 400;
        data_size = 400*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_v, data_size, s_value);
        }
        return;
    }
    if (name == "poissongroup._spikespace") {
        var_size = 785;
        data_size = 785*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_poissongroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_poissongroup__spikespace, data_size, s_value);
        }
        return;
    }
    // dynamic arrays (1d)
    if (name == "synapses_1.delay") {
        var_size = _dynamic_array_synapses_1_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_1.w_ie") {
        var_size = _dynamic_array_synapses_1_w_ie.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_w_ie[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_w_ie[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.Apost1_ee") {
        var_size = _dynamic_array_synapses_2_Apost1_ee.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_Apost1_ee[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_Apost1_ee[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.Apost2_ee") {
        var_size = _dynamic_array_synapses_2_Apost2_ee.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_Apost2_ee[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_Apost2_ee[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.Apost2_prev_ee") {
        var_size = _dynamic_array_synapses_2_Apost2_prev_ee.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_Apost2_prev_ee[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_Apost2_prev_ee[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.Apre_ee") {
        var_size = _dynamic_array_synapses_2_Apre_ee.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_Apre_ee[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_Apre_ee[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.delay") {
        var_size = _dynamic_array_synapses_2_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.delay") {
        var_size = _dynamic_array_synapses_2_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.lastupdate") {
        var_size = _dynamic_array_synapses_2_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.w_ee") {
        var_size = _dynamic_array_synapses_2_w_ee.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_w_ee[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_w_ee[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.delay") {
        var_size = _dynamic_array_synapses_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.w_ei") {
        var_size = _dynamic_array_synapses_w_ei.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_w_ei[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_w_ei[0], data_size, s_value);
        }
        return;
    }
    if (name == "_timedarray.values") {
        var_size = 7840000;
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _timedarray_values, var_size, (double)atof(s_value.c_str()));


        } else {
            // set from file
            set_variable_from_file(name, _timedarray_values, data_size, s_value);
        }
        return;
    }
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
}
//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_1__spikespace;
const int _num__array_neurongroup_1__spikespace = 401;
double * _array_neurongroup_1_g_e;
const int _num__array_neurongroup_1_g_e = 400;
double * _array_neurongroup_1_g_i;
const int _num__array_neurongroup_1_g_i = 400;
int32_t * _array_neurongroup_1_i;
const int _num__array_neurongroup_1_i = 400;
double * _array_neurongroup_1_lastspike;
const int _num__array_neurongroup_1_lastspike = 400;
char * _array_neurongroup_1_not_refractory;
const int _num__array_neurongroup_1_not_refractory = 400;
double * _array_neurongroup_1_v;
const int _num__array_neurongroup_1_v = 400;
int32_t * _array_neurongroup__spikespace;
const int _num__array_neurongroup__spikespace = 401;
double * _array_neurongroup_g_e;
const int _num__array_neurongroup_g_e = 400;
double * _array_neurongroup_g_i;
const int _num__array_neurongroup_g_i = 400;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 400;
double * _array_neurongroup_lastspike;
const int _num__array_neurongroup_lastspike = 400;
char * _array_neurongroup_not_refractory;
const int _num__array_neurongroup_not_refractory = 400;
double * _array_neurongroup_theta;
const int _num__array_neurongroup_theta = 400;
double * _array_neurongroup_v;
const int _num__array_neurongroup_v = 400;
int32_t * _array_poissongroup__spikespace;
const int _num__array_poissongroup__spikespace = 785;
int32_t * _array_poissongroup_i;
const int _num__array_poissongroup_i = 784;
int32_t * _array_spikemonitor__source_idx;
const int _num__array_spikemonitor__source_idx = 400;
int32_t * _array_spikemonitor_count;
const int _num__array_spikemonitor_count = 400;
int32_t * _array_spikemonitor_N;
const int _num__array_spikemonitor_N = 1;
int32_t * _array_synapses_1_N;
const int _num__array_synapses_1_N = 1;
int32_t * _array_synapses_2_N;
const int _num__array_synapses_2_N = 1;
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_spikemonitor_i;
std::vector<double> _dynamic_array_spikemonitor_t;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
std::vector<double> _dynamic_array_synapses_1_delay;
std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
std::vector<double> _dynamic_array_synapses_1_w_ie;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_pre;
std::vector<double> _dynamic_array_synapses_2_Apost1_ee;
std::vector<double> _dynamic_array_synapses_2_Apost2_ee;
std::vector<double> _dynamic_array_synapses_2_Apost2_prev_ee;
std::vector<double> _dynamic_array_synapses_2_Apre_ee;
std::vector<double> _dynamic_array_synapses_2_delay;
std::vector<double> _dynamic_array_synapses_2_delay_1;
std::vector<double> _dynamic_array_synapses_2_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_2_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_2_N_outgoing;
std::vector<double> _dynamic_array_synapses_2_w_ee;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
std::vector<double> _dynamic_array_synapses_w_ei;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
double * _timedarray_values;
const int _num__timedarray_values = 7840000;

//////////////// synapses /////////////////
// synapses
SynapticPathway synapses_pre(
		_dynamic_array_synapses__synaptic_pre,
		0, 400);
// synapses_1
SynapticPathway synapses_1_pre(
		_dynamic_array_synapses_1__synaptic_pre,
		0, 400);
// synapses_2
SynapticPathway synapses_2_post(
		_dynamic_array_synapses_2__synaptic_post,
		0, 400);
SynapticPathway synapses_2_pre(
		_dynamic_array_synapses_2__synaptic_pre,
		0, 784);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_neurongroup_1__spikespace = new int32_t[401];
    
	for(int i=0; i<401; i++) _array_neurongroup_1__spikespace[i] = 0;

	_array_neurongroup_1_g_e = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_1_g_e[i] = 0;

	_array_neurongroup_1_g_i = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_1_g_i[i] = 0;

	_array_neurongroup_1_i = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_1_i[i] = 0;

	_array_neurongroup_1_lastspike = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_1_lastspike[i] = 0;

	_array_neurongroup_1_not_refractory = new char[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_1_not_refractory[i] = 0;

	_array_neurongroup_1_v = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_1_v[i] = 0;

	_array_neurongroup__spikespace = new int32_t[401];
    
	for(int i=0; i<401; i++) _array_neurongroup__spikespace[i] = 0;

	_array_neurongroup_g_e = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_g_e[i] = 0;

	_array_neurongroup_g_i = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_g_i[i] = 0;

	_array_neurongroup_i = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_i[i] = 0;

	_array_neurongroup_lastspike = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_lastspike[i] = 0;

	_array_neurongroup_not_refractory = new char[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_not_refractory[i] = 0;

	_array_neurongroup_theta = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_theta[i] = 0;

	_array_neurongroup_v = new double[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_v[i] = 0;

	_array_poissongroup__spikespace = new int32_t[785];
    
	for(int i=0; i<785; i++) _array_poissongroup__spikespace[i] = 0;

	_array_poissongroup_i = new int32_t[784];
    
	for(int i=0; i<784; i++) _array_poissongroup_i[i] = 0;

	_array_spikemonitor__source_idx = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_spikemonitor__source_idx[i] = 0;

	_array_spikemonitor_count = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_spikemonitor_count[i] = 0;

	_array_spikemonitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;

	_array_synapses_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;

	_array_synapses_2_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;

	_array_synapses_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_N[i] = 0;


	// Arrays initialized to an "arange"
	_array_neurongroup_1_i = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_1_i[i] = 0 + i;

	_array_neurongroup_i = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_neurongroup_i[i] = 0 + i;

	_array_poissongroup_i = new int32_t[784];
    
	for(int i=0; i<784; i++) _array_poissongroup_i[i] = 0 + i;

	_array_spikemonitor__source_idx = new int32_t[400];
    
	for(int i=0; i<400; i++) _array_spikemonitor__source_idx[i] = 0 + i;


	// static arrays
	_timedarray_values = new double[7840000];

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_timedarray_values;
	f_timedarray_values.open("static_arrays/_timedarray_values", ios::in | ios::binary);
	if(f_timedarray_values.is_open())
	{
		f_timedarray_values.read(reinterpret_cast<char*>(_timedarray_values), 7840000*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _timedarray_values." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open(results_dir + "_array_defaultclock_dt_1978099143", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open(results_dir + "_array_defaultclock_t_2669362164", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open(results_dir + "_array_defaultclock_timestep_144223508", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup_1__spikespace;
	outfile__array_neurongroup_1__spikespace.open(results_dir + "_array_neurongroup_1__spikespace_3155027917", ios::binary | ios::out);
	if(outfile__array_neurongroup_1__spikespace.is_open())
	{
		outfile__array_neurongroup_1__spikespace.write(reinterpret_cast<char*>(_array_neurongroup_1__spikespace), 401*sizeof(_array_neurongroup_1__spikespace[0]));
		outfile__array_neurongroup_1__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_1_g_e;
	outfile__array_neurongroup_1_g_e.open(results_dir + "_array_neurongroup_1_g_e_450444230", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_g_e.is_open())
	{
		outfile__array_neurongroup_1_g_e.write(reinterpret_cast<char*>(_array_neurongroup_1_g_e), 400*sizeof(_array_neurongroup_1_g_e[0]));
		outfile__array_neurongroup_1_g_e.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_g_e." << endl;
	}
	ofstream outfile__array_neurongroup_1_g_i;
	outfile__array_neurongroup_1_g_i.open(results_dir + "_array_neurongroup_1_g_i_326072301", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_g_i.is_open())
	{
		outfile__array_neurongroup_1_g_i.write(reinterpret_cast<char*>(_array_neurongroup_1_g_i), 400*sizeof(_array_neurongroup_1_g_i[0]));
		outfile__array_neurongroup_1_g_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_g_i." << endl;
	}
	ofstream outfile__array_neurongroup_1_i;
	outfile__array_neurongroup_1_i.open(results_dir + "_array_neurongroup_1_i_3674354357", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_i.is_open())
	{
		outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 400*sizeof(_array_neurongroup_1_i[0]));
		outfile__array_neurongroup_1_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
	}
	ofstream outfile__array_neurongroup_1_lastspike;
	outfile__array_neurongroup_1_lastspike.open(results_dir + "_array_neurongroup_1_lastspike_1163579662", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_lastspike.is_open())
	{
		outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 400*sizeof(_array_neurongroup_1_lastspike[0]));
		outfile__array_neurongroup_1_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_1_not_refractory;
	outfile__array_neurongroup_1_not_refractory.open(results_dir + "_array_neurongroup_1_not_refractory_897855399", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_not_refractory.is_open())
	{
		outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 400*sizeof(_array_neurongroup_1_not_refractory[0]));
		outfile__array_neurongroup_1_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_1_v;
	outfile__array_neurongroup_1_v.open(results_dir + "_array_neurongroup_1_v_1443512128", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_v.is_open())
	{
		outfile__array_neurongroup_1_v.write(reinterpret_cast<char*>(_array_neurongroup_1_v), 400*sizeof(_array_neurongroup_1_v[0]));
		outfile__array_neurongroup_1_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_v." << endl;
	}
	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open(results_dir + "_array_neurongroup__spikespace_3522821529", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 401*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_g_e;
	outfile__array_neurongroup_g_e.open(results_dir + "_array_neurongroup_g_e_3129289884", ios::binary | ios::out);
	if(outfile__array_neurongroup_g_e.is_open())
	{
		outfile__array_neurongroup_g_e.write(reinterpret_cast<char*>(_array_neurongroup_g_e), 400*sizeof(_array_neurongroup_g_e[0]));
		outfile__array_neurongroup_g_e.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_g_e." << endl;
	}
	ofstream outfile__array_neurongroup_g_i;
	outfile__array_neurongroup_g_i.open(results_dir + "_array_neurongroup_g_i_3006488759", ios::binary | ios::out);
	if(outfile__array_neurongroup_g_i.is_open())
	{
		outfile__array_neurongroup_g_i.write(reinterpret_cast<char*>(_array_neurongroup_g_i), 400*sizeof(_array_neurongroup_g_i[0]));
		outfile__array_neurongroup_g_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_g_i." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open(results_dir + "_array_neurongroup_i_2649026944", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 400*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_lastspike;
	outfile__array_neurongroup_lastspike.open(results_dir + "_array_neurongroup_lastspike_1647074423", ios::binary | ios::out);
	if(outfile__array_neurongroup_lastspike.is_open())
	{
		outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 400*sizeof(_array_neurongroup_lastspike[0]));
		outfile__array_neurongroup_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_not_refractory;
	outfile__array_neurongroup_not_refractory.open(results_dir + "_array_neurongroup_not_refractory_1422681464", ios::binary | ios::out);
	if(outfile__array_neurongroup_not_refractory.is_open())
	{
		outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 400*sizeof(_array_neurongroup_not_refractory[0]));
		outfile__array_neurongroup_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_theta;
	outfile__array_neurongroup_theta.open(results_dir + "_array_neurongroup_theta_2511323641", ios::binary | ios::out);
	if(outfile__array_neurongroup_theta.is_open())
	{
		outfile__array_neurongroup_theta.write(reinterpret_cast<char*>(_array_neurongroup_theta), 400*sizeof(_array_neurongroup_theta[0]));
		outfile__array_neurongroup_theta.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_theta." << endl;
	}
	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open(results_dir + "_array_neurongroup_v_283966581", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 400*sizeof(_array_neurongroup_v[0]));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}
	ofstream outfile__array_poissongroup__spikespace;
	outfile__array_poissongroup__spikespace.open(results_dir + "_array_poissongroup__spikespace_1019000416", ios::binary | ios::out);
	if(outfile__array_poissongroup__spikespace.is_open())
	{
		outfile__array_poissongroup__spikespace.write(reinterpret_cast<char*>(_array_poissongroup__spikespace), 785*sizeof(_array_poissongroup__spikespace[0]));
		outfile__array_poissongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_poissongroup__spikespace." << endl;
	}
	ofstream outfile__array_poissongroup_i;
	outfile__array_poissongroup_i.open(results_dir + "_array_poissongroup_i_1277690444", ios::binary | ios::out);
	if(outfile__array_poissongroup_i.is_open())
	{
		outfile__array_poissongroup_i.write(reinterpret_cast<char*>(_array_poissongroup_i), 784*sizeof(_array_poissongroup_i[0]));
		outfile__array_poissongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_poissongroup_i." << endl;
	}
	ofstream outfile__array_spikemonitor__source_idx;
	outfile__array_spikemonitor__source_idx.open(results_dir + "_array_spikemonitor__source_idx_1477951789", ios::binary | ios::out);
	if(outfile__array_spikemonitor__source_idx.is_open())
	{
		outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 400*sizeof(_array_spikemonitor__source_idx[0]));
		outfile__array_spikemonitor__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_count;
	outfile__array_spikemonitor_count.open(results_dir + "_array_spikemonitor_count_598337445", ios::binary | ios::out);
	if(outfile__array_spikemonitor_count.is_open())
	{
		outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 400*sizeof(_array_spikemonitor_count[0]));
		outfile__array_spikemonitor_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
	}
	ofstream outfile__array_spikemonitor_N;
	outfile__array_spikemonitor_N.open(results_dir + "_array_spikemonitor_N_225734567", ios::binary | ios::out);
	if(outfile__array_spikemonitor_N.is_open())
	{
		outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
		outfile__array_spikemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
	}
	ofstream outfile__array_synapses_1_N;
	outfile__array_synapses_1_N.open(results_dir + "_array_synapses_1_N_1771729519", ios::binary | ios::out);
	if(outfile__array_synapses_1_N.is_open())
	{
		outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(_array_synapses_1_N[0]));
		outfile__array_synapses_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_1_N." << endl;
	}
	ofstream outfile__array_synapses_2_N;
	outfile__array_synapses_2_N.open(results_dir + "_array_synapses_2_N_1809632310", ios::binary | ios::out);
	if(outfile__array_synapses_2_N.is_open())
	{
		outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(_array_synapses_2_N[0]));
		outfile__array_synapses_2_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_2_N." << endl;
	}
	ofstream outfile__array_synapses_N;
	outfile__array_synapses_N.open(results_dir + "_array_synapses_N_483293785", ios::binary | ios::out);
	if(outfile__array_synapses_N.is_open())
	{
		outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
		outfile__array_synapses_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N." << endl;
	}

	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open(results_dir + "_dynamic_array_spikemonitor_i_1976709050", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
		    outfile__dynamic_array_spikemonitor_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open(results_dir + "_dynamic_array_spikemonitor_t_383009635", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
		    outfile__dynamic_array_spikemonitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_post;
	outfile__dynamic_array_synapses_1__synaptic_post.open(results_dir + "_dynamic_array_synapses_1__synaptic_post_1999337987", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_post[0]), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(_dynamic_array_synapses_1__synaptic_post[0]));
		    outfile__dynamic_array_synapses_1__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
	outfile__dynamic_array_synapses_1__synaptic_pre.open(results_dir + "_dynamic_array_synapses_1__synaptic_pre_681065502", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_pre[0]), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(_dynamic_array_synapses_1__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_1__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_delay;
	outfile__dynamic_array_synapses_1_delay.open(results_dir + "_dynamic_array_synapses_1_delay_2373823482", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_delay.is_open())
	{
        if (! _dynamic_array_synapses_1_delay.empty() )
        {
			outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay[0]), _dynamic_array_synapses_1_delay.size()*sizeof(_dynamic_array_synapses_1_delay[0]));
		    outfile__dynamic_array_synapses_1_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_incoming;
	outfile__dynamic_array_synapses_1_N_incoming.open(results_dir + "_dynamic_array_synapses_1_N_incoming_3469555706", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_1_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_incoming[0]), _dynamic_array_synapses_1_N_incoming.size()*sizeof(_dynamic_array_synapses_1_N_incoming[0]));
		    outfile__dynamic_array_synapses_1_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_outgoing;
	outfile__dynamic_array_synapses_1_N_outgoing.open(results_dir + "_dynamic_array_synapses_1_N_outgoing_3922806560", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_1_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_outgoing[0]), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(_dynamic_array_synapses_1_N_outgoing[0]));
		    outfile__dynamic_array_synapses_1_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_w_ie;
	outfile__dynamic_array_synapses_1_w_ie.open(results_dir + "_dynamic_array_synapses_1_w_ie_1803999019", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_w_ie.is_open())
	{
        if (! _dynamic_array_synapses_1_w_ie.empty() )
        {
			outfile__dynamic_array_synapses_1_w_ie.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_w_ie[0]), _dynamic_array_synapses_1_w_ie.size()*sizeof(_dynamic_array_synapses_1_w_ie[0]));
		    outfile__dynamic_array_synapses_1_w_ie.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_w_ie." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2__synaptic_post;
	outfile__dynamic_array_synapses_2__synaptic_post.open(results_dir + "_dynamic_array_synapses_2__synaptic_post_1591987953", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_2__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_post[0]), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(_dynamic_array_synapses_2__synaptic_post[0]));
		    outfile__dynamic_array_synapses_2__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
	outfile__dynamic_array_synapses_2__synaptic_pre.open(results_dir + "_dynamic_array_synapses_2__synaptic_pre_971331175", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_2__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_pre[0]), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(_dynamic_array_synapses_2__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_2__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_Apost1_ee;
	outfile__dynamic_array_synapses_2_Apost1_ee.open(results_dir + "_dynamic_array_synapses_2_Apost1_ee_4239817754", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_Apost1_ee.is_open())
	{
        if (! _dynamic_array_synapses_2_Apost1_ee.empty() )
        {
			outfile__dynamic_array_synapses_2_Apost1_ee.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_Apost1_ee[0]), _dynamic_array_synapses_2_Apost1_ee.size()*sizeof(_dynamic_array_synapses_2_Apost1_ee[0]));
		    outfile__dynamic_array_synapses_2_Apost1_ee.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_Apost1_ee." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_Apost2_ee;
	outfile__dynamic_array_synapses_2_Apost2_ee.open(results_dir + "_dynamic_array_synapses_2_Apost2_ee_3993228276", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_Apost2_ee.is_open())
	{
        if (! _dynamic_array_synapses_2_Apost2_ee.empty() )
        {
			outfile__dynamic_array_synapses_2_Apost2_ee.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_Apost2_ee[0]), _dynamic_array_synapses_2_Apost2_ee.size()*sizeof(_dynamic_array_synapses_2_Apost2_ee[0]));
		    outfile__dynamic_array_synapses_2_Apost2_ee.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_Apost2_ee." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_Apost2_prev_ee;
	outfile__dynamic_array_synapses_2_Apost2_prev_ee.open(results_dir + "_dynamic_array_synapses_2_Apost2_prev_ee_4224553539", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_Apost2_prev_ee.is_open())
	{
        if (! _dynamic_array_synapses_2_Apost2_prev_ee.empty() )
        {
			outfile__dynamic_array_synapses_2_Apost2_prev_ee.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_Apost2_prev_ee[0]), _dynamic_array_synapses_2_Apost2_prev_ee.size()*sizeof(_dynamic_array_synapses_2_Apost2_prev_ee[0]));
		    outfile__dynamic_array_synapses_2_Apost2_prev_ee.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_Apost2_prev_ee." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_Apre_ee;
	outfile__dynamic_array_synapses_2_Apre_ee.open(results_dir + "_dynamic_array_synapses_2_Apre_ee_1354619856", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_Apre_ee.is_open())
	{
        if (! _dynamic_array_synapses_2_Apre_ee.empty() )
        {
			outfile__dynamic_array_synapses_2_Apre_ee.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_Apre_ee[0]), _dynamic_array_synapses_2_Apre_ee.size()*sizeof(_dynamic_array_synapses_2_Apre_ee[0]));
		    outfile__dynamic_array_synapses_2_Apre_ee.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_Apre_ee." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_delay;
	outfile__dynamic_array_synapses_2_delay.open(results_dir + "_dynamic_array_synapses_2_delay_3163926887", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_delay.is_open())
	{
        if (! _dynamic_array_synapses_2_delay.empty() )
        {
			outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_delay[0]), _dynamic_array_synapses_2_delay.size()*sizeof(_dynamic_array_synapses_2_delay[0]));
		    outfile__dynamic_array_synapses_2_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_delay_1;
	outfile__dynamic_array_synapses_2_delay_1.open(results_dir + "_dynamic_array_synapses_2_delay_1_3154022833", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_delay_1.is_open())
	{
        if (! _dynamic_array_synapses_2_delay_1.empty() )
        {
			outfile__dynamic_array_synapses_2_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_delay_1[0]), _dynamic_array_synapses_2_delay_1.size()*sizeof(_dynamic_array_synapses_2_delay_1[0]));
		    outfile__dynamic_array_synapses_2_delay_1.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_delay_1." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_lastupdate;
	outfile__dynamic_array_synapses_2_lastupdate.open(results_dir + "_dynamic_array_synapses_2_lastupdate_551200724", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_lastupdate.is_open())
	{
        if (! _dynamic_array_synapses_2_lastupdate.empty() )
        {
			outfile__dynamic_array_synapses_2_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_lastupdate[0]), _dynamic_array_synapses_2_lastupdate.size()*sizeof(_dynamic_array_synapses_2_lastupdate[0]));
		    outfile__dynamic_array_synapses_2_lastupdate.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_N_incoming;
	outfile__dynamic_array_synapses_2_N_incoming.open(results_dir + "_dynamic_array_synapses_2_N_incoming_3109283082", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_2_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_incoming[0]), _dynamic_array_synapses_2_N_incoming.size()*sizeof(_dynamic_array_synapses_2_N_incoming[0]));
		    outfile__dynamic_array_synapses_2_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_N_outgoing;
	outfile__dynamic_array_synapses_2_N_outgoing.open(results_dir + "_dynamic_array_synapses_2_N_outgoing_2656015824", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_2_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_outgoing[0]), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(_dynamic_array_synapses_2_N_outgoing[0]));
		    outfile__dynamic_array_synapses_2_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_2_w_ee;
	outfile__dynamic_array_synapses_2_w_ee.open(results_dir + "_dynamic_array_synapses_2_w_ee_1101523593", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_2_w_ee.is_open())
	{
        if (! _dynamic_array_synapses_2_w_ee.empty() )
        {
			outfile__dynamic_array_synapses_2_w_ee.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_w_ee[0]), _dynamic_array_synapses_2_w_ee.size()*sizeof(_dynamic_array_synapses_2_w_ee[0]));
		    outfile__dynamic_array_synapses_2_w_ee.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_2_w_ee." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_post;
	outfile__dynamic_array_synapses__synaptic_post.open(results_dir + "_dynamic_array_synapses__synaptic_post_1801389495", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
		    outfile__dynamic_array_synapses__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open(results_dir + "_dynamic_array_synapses__synaptic_pre_814148175", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
		    outfile__dynamic_array_synapses__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_delay;
	outfile__dynamic_array_synapses_delay.open(results_dir + "_dynamic_array_synapses_delay_3246960869", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_delay.is_open())
	{
        if (! _dynamic_array_synapses_delay.empty() )
        {
			outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
		    outfile__dynamic_array_synapses_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_incoming;
	outfile__dynamic_array_synapses_N_incoming.open(results_dir + "_dynamic_array_synapses_N_incoming_1151751685", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
		    outfile__dynamic_array_synapses_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_outgoing;
	outfile__dynamic_array_synapses_N_outgoing.open(results_dir + "_dynamic_array_synapses_N_outgoing_1673144031", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
		    outfile__dynamic_array_synapses_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_w_ei;
	outfile__dynamic_array_synapses_w_ei.open(results_dir + "_dynamic_array_synapses_w_ei_1385076013", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_w_ei.is_open())
	{
        if (! _dynamic_array_synapses_w_ei.empty() )
        {
			outfile__dynamic_array_synapses_w_ei.write(reinterpret_cast<char*>(&_dynamic_array_synapses_w_ei[0]), _dynamic_array_synapses_w_ei.size()*sizeof(_dynamic_array_synapses_w_ei[0]));
		    outfile__dynamic_array_synapses_w_ei.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_w_ei." << endl;
	}

	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
}

void _dealloc_arrays()
{
	using namespace brian;


	// static arrays
	if(_timedarray_values!=0)
	{
		delete [] _timedarray_values;
		_timedarray_values = 0;
	}
}

