#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_poissongroup_spike_thresholder_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/before_run_synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_2_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_2_post_codeobject.h"
#include "code_objects/synapses_2_post_push_spikes.h"
#include "code_objects/before_run_synapses_2_post_push_spikes.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/before_run_synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>


        std::string _format_time(float time_in_s)
        {
            float divisors[] = {24*60*60, 60*60, 60, 1};
            char letters[] = {'d', 'h', 'm', 's'};
            float remaining = time_in_s;
            std::string text = "";
            int time_to_represent;
            for (int i =0; i < sizeof(divisors)/sizeof(float); i++)
            {
                time_to_represent = int(remaining / divisors[i]);
                remaining -= time_to_represent * divisors[i];
                if (time_to_represent > 0 || text.length())
                {
                    if(text.length() > 0)
                    {
                        text += " ";
                    }
                    text += (std::to_string(time_to_represent)+letters[i]);
                }
            }
            //less than one second
            if(text.length() == 0)
            {
                text = "< 1s";
            }
            return text;
        }
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                std::cout << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                std::cout << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << _format_time(elapsed);
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    std::cout << ", estimated " << _format_time(remaining) << " remaining.";
                }
            }

            std::cout << std::endl << std::flush;
        }
        


void set_from_command_line(const std::vector<std::string> args)
{
    for (const auto& arg : args) {
		// Split into two parts
		size_t equal_sign = arg.find("=");
		auto name = arg.substr(0, equal_sign);
		auto value = arg.substr(equal_sign + 1, arg.length());
		brian::set_variable_by_name(name, value);
	}
}
int main(int argc, char **argv)
{
	std::vector<std::string> args(argv + 1, argv + argc);
	if (args.size() >=2 && args[0] == "--results_dir")
	{
		brian::results_dir = args[1];
		#ifdef DEBUG
		std::cout << "Setting results dir to '" << brian::results_dir << "'" << std::endl;
		#endif
		args.erase(args.begin(), args.begin()+2);
	}
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        
                        
                        for(int i=0; i<_num__array_neurongroup_lastspike; i++)
                        {
                            _array_neurongroup_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_not_refractory; i++)
                        {
                            _array_neurongroup_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_lastspike; i++)
                        {
                            _array_neurongroup_1_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_not_refractory; i++)
                        {
                            _array_neurongroup_1_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_v; i++)
                        {
                            _array_neurongroup_v[i] = - 0.10500000000000001;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_v; i++)
                        {
                            _array_neurongroup_1_v[i] = - 0.1;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_theta; i++)
                        {
                            _array_neurongroup_theta[i] = 0.02;
                        }
                        
        _run_synapses_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_w_ei.size(); i++)
                        {
                            _dynamic_array_synapses_w_ei[i] = 10.4;
                        }
                        
        _run_synapses_1_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_1_w_ie.size(); i++)
                        {
                            _dynamic_array_synapses_1_w_ie[i] = 17;
                        }
                        
        _run_synapses_2_synapses_create_generator_codeobject();
        _run_synapses_2_group_variable_set_conditional_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_2_delay.size(); i++)
                        {
                            _dynamic_array_synapses_2_delay[i] = 0.01;
                        }
                        
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _before_run_synapses_1_pre_push_spikes();
        _before_run_synapses_2_pre_push_spikes();
        _before_run_synapses_pre_push_spikes();
        _before_run_synapses_2_post_push_spikes();
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_poissongroup_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_2_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_2_post_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_spike_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_spike_resetter_codeobject);
        set_from_command_line(args);
        magicnetwork.run(5000.0, report_progress, 10.0);
        _after_run_neurongroup_1_spike_thresholder_codeobject();
        _after_run_neurongroup_spike_thresholder_codeobject();
        _after_run_poissongroup_spike_thresholder_codeobject();
        #ifdef DEBUG
        _debugmsg_spikemonitor_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_post_codeobject();
        #endif

	}
        

	brian_end();
        

	return 0;
}