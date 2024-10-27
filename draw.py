import argparse
import pickle
import numpy as np
from brian2 import *
from pathlib import Path


def draw_heatmap(model, spike_counts, img_name):
    spike_counts = np.array(spike_counts)
    spike_counts_grid = spike_counts.reshape(model.layout, model.layout)

    plt.clf()
    # Plotting the spike counts in a grid
    plt.figure(figsize=(12, 12))
    plt.imshow(spike_counts_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Spike Count')
    plt.title(f'{img_name}')
    plt.xlabel('Neuron X')
    plt.ylabel('Neuron Y')

    # Optional: annotate each square with the spike count
    for i in range(model.layout):
        for j in range(model.layout):
            plt.text(j, i, int(spike_counts_grid[i, j]), ha='center', va='center', color='white')
    plt.savefig(f"{img_name}.png")
    # plt.show()


def draw_weights(model, synapse, img_name):
    fig, ax = plt.subplots(model.layout, model.layout, figsize=(40, 40))
    dim = int(np.sqrt(model.args.population_exc))

    weights = np.zeros((dim, dim), dtype=float)
    prev_post_idx = 0
    syn_count = len(synapse["w_ee"])
    for syn_idx in range(syn_count):
        pre_idx = synapse["i"][syn_idx]
        post_idx = synapse["j"][syn_idx]
        if prev_post_idx == post_idx and syn_idx != syn_count - 1:
            weights[pre_idx // dim, pre_idx % dim] = synapse["w_ee"][syn_idx]
        else:
            row = prev_post_idx // model.layout
            col = prev_post_idx % model.layout

            ax[row, col].imshow(weights, vmin=0, vmax=1)
            ax[row, col].axis('off')
            weights = np.zeros((dim, dim), dtype=float)
            weights[pre_idx // 28, pre_idx % 28] = synapse["w_ee"][syn_idx]
            prev_post_idx = post_idx

    plt.tight_layout()
    plt.savefig(f"{img_name}.png")


def draw_accuracies(model, accuracies):
    if model.args.test_phase:
        run_label = "test"
    else:
        run_label = "training"
    # iteration is x label of graph
    iteration = [run_cnt * model.args.image_count + img_idx for run_cnt in range(model.args.run_count) for img_idx in
                 range(model.args.acc_update_interval, model.args.image_count + 1, model.args.acc_update_interval)]

    plt.figure(100)
    plt.plot(iteration, accuracies)
    plt.title(f'Accuracy change over iterations for {run_label} phase')
    plt.xlabel("Iteration Count")
    plt.ylabel("Accuracy % ")
    plt.grid(True)
    plt.savefig(f'{model.run_path}/{run_label}_accuracy_graph.png')


def load_data(path: str):
    with open(f"{path}.pickle", 'rb') as f:
        d = pickle.load(f)
    return d


def get_folder_names(path: str):
    folder_path = Path(path)
    folder_names = [f.stem for f in folder_path.iterdir() if f.is_dir()]
    return folder_names


def get_file_names(path: str):
    folder_path = Path(path)
    file_names = [f.stem for f in folder_path.iterdir() if f.is_file()]
    return file_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None, help="run name")
    parser.add_argument('--test_phase', action='store_true', help="Set this flag to indicate test_phase")

    args = parser.parse_args()

    mode = "test" if args.test_phase else "train"

    model_path = f"runs/{args.run_name}/model_dump/model_{mode}"
    model = load_data(model_path)

    acc_data_path = f"{model.acc_dump_path}/accuracies_{mode}"
    accuracies = load_data(acc_data_path)
    draw_accuracies(model, accuracies)

    spike_mon_dump_path = f"{model.run_path}/spike_mon_dump/{mode}"
    spike_mon_dirs = get_folder_names(spike_mon_dump_path)
    for spike_mon_dir in spike_mon_dirs:
        file_names = get_file_names(f"{spike_mon_dump_path}/{spike_mon_dir}")
        for file_name in file_names:
            path = f"{spike_mon_dump_path}/{spike_mon_dir}/{file_name}"
            spike_mon_data = load_data(path)
            draw_heatmap(model, spike_mon_data, path)

    weight_dump_path = f"{model.run_path}/weight_dump"
    weight_dirs = get_folder_names(weight_dump_path)
    for weight_dir in weight_dirs:
        file_names = get_file_names(f"{weight_dump_path}/{weight_dir}")
        for file_name in file_names:
            path = f"{weight_dump_path}/{weight_dir}/{file_name}"
            syn_data = load_data(path)
            draw_weights(model, syn_data, path)
