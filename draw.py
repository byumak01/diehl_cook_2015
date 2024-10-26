from brian2 import *

def draw_heatmap(model, spike_counts, img_name):
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
    plt.savefig(f"{model.run_path}/{img_name}.png")
    # plt.show()


def draw_weights(model, synapse, img_name):
    fig, ax = plt.subplots(model.layout, model.layout, figsize=(40, 40))
    dim = int(sqrt(model.args.population_exc))
    for post_idx in range(model.args.population_exc):
        pre_indices_for_current_post = sort(receptive_field_for_exc(model, post_idx))
        weights = np.zeros((dim, dim), dtype=float)
        weights[pre_indices_for_current_post//28, pre_indices_for_current_post%28] = synapse.w_ee[pre_indices_for_current_post, post_idx]

        row = post_idx // model.layout
        col = post_idx % model.layout

        ax[row, col].imshow(weights, vmin=0, vmax=1)
        ax[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(f"{model.run_path}/{img_name}.png")


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