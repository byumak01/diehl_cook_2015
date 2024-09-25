PROGRAM = main

SRCS = brianlib/randomkit/randomkit.c code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.cpp code_objects/after_run_neurongroup_spike_thresholder_codeobject.cpp code_objects/after_run_poissongroup_spike_thresholder_codeobject.cpp code_objects/before_run_synapses_1_pre_push_spikes.cpp code_objects/before_run_synapses_2_post_push_spikes.cpp code_objects/before_run_synapses_2_pre_push_spikes.cpp code_objects/before_run_synapses_pre_push_spikes.cpp code_objects/neurongroup_1_spike_resetter_codeobject.cpp code_objects/neurongroup_1_spike_thresholder_codeobject.cpp code_objects/neurongroup_1_stateupdater_codeobject.cpp code_objects/neurongroup_spike_resetter_codeobject.cpp code_objects/neurongroup_spike_thresholder_codeobject.cpp code_objects/neurongroup_stateupdater_codeobject.cpp code_objects/poissongroup_spike_thresholder_codeobject.cpp code_objects/spikemonitor_codeobject.cpp code_objects/synapses_1_pre_codeobject.cpp code_objects/synapses_1_pre_push_spikes.cpp code_objects/synapses_1_synapses_create_generator_codeobject.cpp code_objects/synapses_2_group_variable_set_conditional_codeobject.cpp code_objects/synapses_2_post_codeobject.cpp code_objects/synapses_2_post_push_spikes.cpp code_objects/synapses_2_pre_codeobject.cpp code_objects/synapses_2_pre_push_spikes.cpp code_objects/synapses_2_synapses_create_generator_codeobject.cpp code_objects/synapses_pre_codeobject.cpp code_objects/synapses_pre_push_spikes.cpp code_objects/synapses_synapses_create_generator_codeobject.cpp main.cpp network.cpp objects.cpp run.cpp synapses_classes.cpp
H_SRCS = brianlib/clocks.h brianlib/common_math.h brianlib/dynamic_array.h code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h code_objects/after_run_neurongroup_spike_thresholder_codeobject.h code_objects/after_run_poissongroup_spike_thresholder_codeobject.h code_objects/before_run_synapses_1_pre_push_spikes.h code_objects/before_run_synapses_2_post_push_spikes.h code_objects/before_run_synapses_2_pre_push_spikes.h code_objects/before_run_synapses_pre_push_spikes.h code_objects/neurongroup_1_spike_resetter_codeobject.h code_objects/neurongroup_1_spike_thresholder_codeobject.h code_objects/neurongroup_1_stateupdater_codeobject.h code_objects/neurongroup_spike_resetter_codeobject.h code_objects/neurongroup_spike_thresholder_codeobject.h code_objects/neurongroup_stateupdater_codeobject.h code_objects/poissongroup_spike_thresholder_codeobject.h code_objects/spikemonitor_codeobject.h code_objects/synapses_1_pre_codeobject.h code_objects/synapses_1_pre_push_spikes.h code_objects/synapses_1_synapses_create_generator_codeobject.h code_objects/synapses_2_group_variable_set_conditional_codeobject.h code_objects/synapses_2_post_codeobject.h code_objects/synapses_2_post_push_spikes.h code_objects/synapses_2_pre_codeobject.h code_objects/synapses_2_pre_push_spikes.h code_objects/synapses_2_synapses_create_generator_codeobject.h code_objects/synapses_pre_codeobject.h code_objects/synapses_pre_push_spikes.h code_objects/synapses_synapses_create_generator_codeobject.h network.h objects.h run.h synapses_classes.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
OPTIMISATIONS = -Ibrianlib/randomkit -I/mnt/c/Users/bymk/Documents/diehl2015/.venv_wsl/include -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11
CXXFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -Lbrianlib/randomkit -L/mnt/c/Users/bymk/Documents/diehl2015/.venv_wsl/lib -Wl,--enable-new-dtags,-rpath,/mnt/c/Users/bymk/Documents/diehl2015/.venv_wsl/lib 
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CXX) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) $< -o $@