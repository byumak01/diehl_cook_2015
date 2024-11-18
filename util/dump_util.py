import pickle, os, csv
from util.syn_util import package_syn_data
from brian2 import *


def ensure_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path: str):
    with open(f"{path}.pickle", 'rb') as f:
        d = pickle.load(f)
    return d


def dump_data(d, folder_path: str, dump_name: str) -> None:
    path = f"{folder_path}/{dump_name}.pickle"
    ensure_path(folder_path)
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def dump_theta_values(model, exc_neuron_groups, dump_time: str):
    for idx, exc_neuron_group in enumerate(exc_neuron_groups):
        print(exc_neuron_group)
        dump_data(exc_neuron_group.theta[:], model.theta_dump_path, f"{dump_time}_theta_ng{idx}_{model.mode}")


def dump_weights(model, synapses: list[Synapses], dump_time: str):
    for idx, synapse in enumerate(synapses):
        package = package_syn_data(synapse)
        dump_data(package, f"{model.weight_dump_path}/ee_syn{idx}_{model.mode}",
                  f"{dump_time}_ee_syn{idx}_{model.mode}")


def write_to_csv(model, accuracy, sim_time, filename='../experimental_runs/experimental_runs.csv'):
    # Get a dictionary of all arguments
    args_dict = vars(model.args)

    # Add the accuracy to the dictionary
    args_dict['accuracy'] = accuracy
    args_dict['sim_time'] = sim_time

    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)

    # Writing to a CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header (parameter names) if file does not exist
        if not file_exists:
            writer.writerow(args_dict.keys())

        # Write parameter values (append to the file)
        writer.writerow(args_dict.values())

    print(f"Parameters and accuracy appended to {filename}")
