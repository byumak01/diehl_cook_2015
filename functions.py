from typing import List
import numpy as np
import os

def load_all_precomputed_images_and_labels(directory, seed=None):
    """
    Loads all precomputed rate-based images and labels into arrays.

    Args:
        directory (str): Path to the dataset folder.
        seed (int, optional): If provided, loads the seeded versions of image rates and labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (image_rates, labels)
    """
    # Construct file paths based on seed
    suffix = f"_seed{seed}" if seed is not None else ""
    images_rates_path = os.path.join(directory, f"images_rates{suffix}.npy")
    labels_path = os.path.join(directory, f"labels{suffix}.npy")

    if not os.path.exists(images_rates_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find required files in {directory}.")

    # Load entire arrays
    images_rates = np.load(images_rates_path)
    labels = np.load(labels_path)

    return images_rates, labels

def load_and_assign_neuron_labels(
    selected_spike_data_path: str,
    spike_data_with_labels_folder_path: str,
    population_exc: int
    ) -> str:
    """
    Loads spike data, assigns labels based on spike activity, and saves the assigned labels.

    :param selected_spike_data_path: Path to the .npz file with spike data and labels.
    :param population_exc: Number of excitatory neurons.
    :return: Path where the assigned labels are saved.
    """
    # Load spike counts and labels from the .npz file
    loaded_data = np.load(selected_spike_data_path)
    print(f'[INFO] Loaded spike data from {selected_spike_data_path}')

    loaded_spike_counts = loaded_data['spike_counts']
    loaded_labels = loaded_data['labels']

    print("loaded_spike_counts len")
    print(len(loaded_spike_counts))
    print("loaded_labels len")
    print(len(loaded_labels))

    # Initialize assigned labels and max average spike counts
    assigned_labels = np.ones(population_exc, dtype=int) * -1
    maximum_average_spike_counts = [0] * population_exc

    # Ensure shapes match
    assert loaded_spike_counts.shape[0] == loaded_labels.shape[0], \
        "Spike counts and labels must match in number of images."

    # Assign labels based on max average spike counts
    for label in range(10):
        current_label_indices = np.where(loaded_labels == label)[0]
        current_label_count = len(current_label_indices)

        if current_label_count > 0:
            total_spike_counts = np.sum(loaded_spike_counts[current_label_indices], axis=0)
            average_spike_counts = total_spike_counts / current_label_count
            for neuron_idx in range(population_exc):
                if average_spike_counts[neuron_idx] > maximum_average_spike_counts[neuron_idx]:
                    maximum_average_spike_counts[neuron_idx] = average_spike_counts[neuron_idx]
                    assigned_labels[neuron_idx] = label

    assigned_labels_path = f"{spike_data_with_labels_folder_path}/assignments_from_training.npy"

    # Save the assigned labels
    np.save(assigned_labels_path, assigned_labels)
    print(f'[INFO] Neuron labels saved at {assigned_labels_path}')

    return assigned_labels_path


def save_synapse_attributes(syn_input_exc, directory, prefix="syn_input_exc"):
    """
    Saves important attributes of a synapse object (e.g., syn_input_exc) to the specified directory.

    Parameters:
    - syn_input_exc: The synapse object to save attributes from.
    - directory: Directory to save the attributes in.
    - prefix: Prefix for filenames (default is "syn_input_exc").
    """
    os.makedirs(directory, exist_ok=True)

    # Save synapse weights (e.g., `w_ee`) and target indices (`j`) if they exist
    if hasattr(syn_input_exc, 'w_ee'):
        np.save(os.path.join(directory, f"{prefix}_weights.npy"), syn_input_exc.w_ee[:])
    if hasattr(syn_input_exc, 'j'):
        np.save(os.path.join(directory, f"{prefix}_indices.npy"), syn_input_exc.j[:])

    print(f"[INFO] Synapse attributes saved in {directory}.")

def save_neuron_group_exc_attributes(neuron_group_exc, directory, prefix="neuron_group_exc"):
    """
    Saves important attributes of a neuron group object (e.g., neuron_group_exc) to the specified directory.

    Parameters:
    - neuron_group_exc: The neuron group object to save attributes from.
    - directory: Directory to save the attributes in.
    - prefix: Prefix for filenames (default is "neuron_group_exc").
    """
    os.makedirs(directory, exist_ok=True)

    # Save attributes like `v`, `theta`, etc., if they exist
    if hasattr(neuron_group_exc, 'v'):
        np.save(os.path.join(directory, f"{prefix}_voltages.npy"), neuron_group_exc.v[:])
    if hasattr(neuron_group_exc, 'theta'):
        np.save(os.path.join(directory, f"{prefix}_theta.npy"), neuron_group_exc.theta[:])

    print(f"[INFO] Neuron group (excitatory) attributes saved in {directory}.")

def save_neuron_group_inh_attributes(neuron_group_inh, directory, prefix="neuron_group_inh"):
    """
    Saves important attributes of a neuron group object (e.g., neuron_group_inh) to the specified directory.

    Parameters:
    - neuron_group_inh: The neuron group object to save attributes from.
    - directory: Directory to save the attributes in.
    - prefix: Prefix for filenames (default is "neuron_group_inh").
    """
    os.makedirs(directory, exist_ok=True)

    # Save attributes like `v`, `g_e`, etc., if they exist
    if hasattr(neuron_group_inh, 'v'):
        np.save(os.path.join(directory, f"{prefix}_voltages.npy"), neuron_group_inh.v[:])
    if hasattr(neuron_group_inh, 'g_e'):
        np.save(os.path.join(directory, f"{prefix}_g_e.npy"), neuron_group_inh.g_e[:])
    if hasattr(neuron_group_inh, 'g_i'):
        np.save(os.path.join(directory, f"{prefix}_g_i.npy"), neuron_group_inh.g_i[:])

    print(f"[INFO] Neuron group (inhibitory) attributes saved in {directory}.")

def save_simulation_state(run_dir, last_image_index, syn_input_exc, neuron_group_exc, neuron_group_inh, save_simulation_state_folder: str = None):

    if save_simulation_state_folder is None:
        save_simulation_state_folder = run_dir
    os.makedirs(save_simulation_state_folder, exist_ok=True)

    save_synapse_attributes(syn_input_exc, save_simulation_state_folder)
    save_neuron_group_exc_attributes(neuron_group_exc, save_simulation_state_folder)
    save_neuron_group_inh_attributes(neuron_group_inh, save_simulation_state_folder)

    final_weights = syn_input_exc.w_ee[:]  # Get the synaptic weights
    np.save(f'{save_simulation_state_folder}/final_synaptic_weights.npy', final_weights)  # Save the weights
    np.save(f'{save_simulation_state_folder}/neuron_group_exc_theta.npy', neuron_group_exc.theta[:])  # Threshold values for excitatory neurons
    print(f"[INFO] Final synaptic weights saved for run: {run_dir}")
    print(f"[INFO] Theta values saved for run: {run_dir}")

    np.save(f'{save_simulation_state_folder}/neuron_group_exc_v.npy', neuron_group_exc.v[:])  # Membrane potentials for excitatory neurons
    np.save(f'{save_simulation_state_folder}/neuron_group_inh_v.npy', neuron_group_inh.v[:])  # Membrane potentials for inhibitory neurons

    # Save the current image index
    np.save(f'{save_simulation_state_folder}/last_image_index.npy', np.array([last_image_index]))
    print(f"[INFO] Simulation state saved after processing {last_image_index+1} images.")


def finalize_prediction_report(prediction_folder_path: str, predicted_labels: list, image_labels_in_loop: list,
                               cumulative_accuracies: list, image_indexes_in_loop: list, start_index: int = 0) -> None:

    """
    Combines image_labels_in_loop, predicted_labels, and cumulative_accuracies into a text file report.

    :param prediction_folder_path: Directory path where the final report will be saved.
    :param predicted_labels: List of predicted labels.
    :param image_labels_in_loop: List of true labels.
    :param cumulative_accuracies: List of cumulative accuracy values.
    :param image_index_in_loop: Index of the image in the loop, to be printed for each entry.
    :param start_index: Starting index for the data selection in image labels.
    """
    # Define the save path
    save_path = f"{prediction_folder_path}/prediction_report.txt"
    tot_ims_seen = [index + 1 for index in image_indexes_in_loop]

    with open(save_path, 'w') as f:
        f.write("Total Images Seen\tImage Index in the Loop\tIndex as in Image Labels\tTrue Label\tPredicted Label\tCorrect Classification\tCumulative Accuracy\n")
        for i, (tot_im_seen, image_idx_in_loop, true_label, pred_label, acc) in enumerate(zip(tot_ims_seen, image_indexes_in_loop, image_labels_in_loop, predicted_labels, cumulative_accuracies)):
            is_correct = true_label == pred_label
            current_index = image_idx_in_loop + start_index
            f.write(f"{tot_im_seen}\t{image_idx_in_loop}\t{current_index}\t{true_label}\t{pred_label}\t{is_correct}\t{acc:.2f}%\n")


def calculate_batch_accuracy(predicted_labels: List[int], true_labels: List[int]) -> float:
    correct_predictions = sum([1 for p, t in zip(predicted_labels, true_labels) if p == t])
    return (correct_predictions / len(predicted_labels)) * 100

def get_batch_accuracies(predicted_labels: List[int], true_labels: List[int], label_predict_range: int) -> List[float]:
    batch_accuracies = []
    for idx in range(0, len(predicted_labels), label_predict_range):
        batch_pred_labels = predicted_labels[idx:idx + label_predict_range]
        batch_true_labels = true_labels[idx:idx + label_predict_range]
        batch_accuracy = calculate_batch_accuracy(batch_pred_labels, batch_true_labels)
        batch_accuracies.append(batch_accuracy)
    return batch_accuracies


def save_report(report_content: List[str], output_dir: str, filename: str = "training_report.txt") -> None:
    report_path = os.path.join(output_dir, filename)
    with open(report_path, 'w') as report_file:
        for line in report_content:
            report_file.write(line + '\n')
    print(f"[INFO] Saved training report to: {report_path}")

def load_and_analyze_training_data(label_predict_range: int, output_dir: str) -> None:
    """
    Loads training data, performs analysis, and saves report content to a text file.

    Parameters:
    - label_predict_range: Number of images per label prediction batch.
    - output_dir: Directory to save analysis results and report.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data files
    predicted_labels = np.load(f'{output_dir}/predicted_labels.npy')
    image_labels = np.load(f'{output_dir}/image_labels_in_loop.npy')
    # image_indexes_in_loop = np.load(f'{output_dir}/image_indexes_in_loop.npy')
    cumulative_accuracies = np.load(f'{output_dir}/cumulative_accuracies.npy')

    # Analysis and report generation
    last_cumulative_accuracy = cumulative_accuracies[-1]
    total_seen_images = len(predicted_labels) + label_predict_range
    print(f"[INFO] Final cumulative accuracy over all seen images: {last_cumulative_accuracy:.2f}%")
    print(f"[INFO] Total images seen for training: {total_seen_images}")

    report_content = [
        f"Final cumulative accuracy over all seen images: {last_cumulative_accuracy:.2f}%",
        f"Total images seen for training: {total_seen_images}"
    ]

    last_batch_pred_labels = predicted_labels[-label_predict_range:]
    last_batch_true_labels = image_labels[-label_predict_range:]
    last_batch_accuracy = calculate_batch_accuracy(last_batch_pred_labels, last_batch_true_labels)
    print(f"[INFO] Accuracy for the last batch of {label_predict_range} images: {last_batch_accuracy:.2f}%")
    report_content.append(f"Accuracy for the last batch of {label_predict_range} images: {last_batch_accuracy:.2f}%")

    batch_accuracies = get_batch_accuracies(predicted_labels, image_labels, label_predict_range)
    for idx, accuracy in enumerate(batch_accuracies, start=1):
        report_content.append(f"Accuracy for batch {idx}: {accuracy:.2f}%")
        print(f"[INFO] Accuracy for batch {idx}: {accuracy:.2f}%")

    np.save(f"{output_dir}/batch_accuracies.npy", np.array(batch_accuracies))
    print(f"[INFO] Saved batch accuracies as .npy file to: {output_dir}")

    # Save report content to a text file
    save_report(report_content, output_dir)


def calculate_accuracy(prediction_folder_path) -> float:
    """
    Calculate accuracy based on true labels and predicted labels, and save to a .txt file.

    :param prediction_folder_path: Directory path where the results will be saved.
    :return: Accuracy as a percentage (float).
    """
    true_labels_path = f'{prediction_folder_path}/image_labels_in_loop.npy'
    true_labels = np.load(true_labels_path)

    predicted_labels_path = f'{prediction_folder_path}/predicted_labels.npy'
    predicted_labels = np.load(predicted_labels_path)

    # Ensure both arrays have the same length
    assert len(true_labels) == len(predicted_labels), "Length of true labels and predicted labels must match."

    # Calculate correct predictions and accuracy
    correct_predictions = np.sum(true_labels == predicted_labels)
    accuracy = (correct_predictions / len(true_labels)) * 100

    # Save results to a .txt file
    save_path = os.path.join(prediction_folder_path, 'accuracy_results.txt')
    with open(save_path, 'w') as f:
        f.write(f'Overall Accuracy: {accuracy:.2f}%\n')
        f.write(f'Total Images: {len(true_labels)}\n')
        f.write(f'Correct Predictions: {correct_predictions}\n')
        f.write(f'Incorrect Predictions: {len(true_labels) - correct_predictions}\n')

    print(f"[INFO] Accuracy results saved to {save_path}")
    return accuracy

def calculate_accuracy_per_label(
    prediction_folder_path: str,
    num_labels: int = 10
) -> np.ndarray:
    """
    Calculate accuracy for each label and save to a .txt file.

    :param prediction_folder_path: Path where the results will be saved.
    :param true_labels_path: Path to true labels file.
    :param predicted_labels_path: Path to predicted labels file.
    :param num_labels: Total number of unique labels (default is 10 for digits).
    :return: Accuracy per label as a numpy array.
    """
    true_labels_path = f'{prediction_folder_path}/image_labels_in_loop.npy'
    true_labels = np.load(true_labels_path)

    predicted_labels_path = f'{prediction_folder_path}/predicted_labels.npy'
    predicted_labels = np.load(predicted_labels_path)

    # Initialize an array to store accuracy per label
    accuracy_per_label = np.zeros(num_labels)

    # Open the file to save accuracy per label
    save_text_path = os.path.join(prediction_folder_path, 'accuracy_per_label.txt')
    with open(save_text_path, 'w') as f:
        for label in range(num_labels):
            # Get indices where the true label is the current label
            label_indices = np.where(true_labels == label)[0]
            if len(label_indices) > 0:
                # Calculate accuracy for this label
                correct_predictions = np.sum(true_labels[label_indices] == predicted_labels[label_indices])
                accuracy = (correct_predictions / len(label_indices)) * 100
                accuracy_per_label[label] = accuracy

                # Write accuracy for each label to the file
                f.write(f'Label {label} Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(label_indices)})\n')

    save_npy_path = os.path.join(prediction_folder_path, 'accuracy_per_label.npy')
    np.save(save_npy_path, accuracy_per_label)
    print(f"[INFO] Accuracy per label results saved to {save_text_path} and {save_npy_path}")

