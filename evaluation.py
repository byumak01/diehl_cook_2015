import typing as ty
import numpy as np

def assign_neurons_to_labels(spike_counts_per_image: ty.List[ty.List[int]], image_labels: np.ndarray, population_exc: int, run_name: str) -> None:
    assigned_labels = np.ones(population_exc, dtype=int) * -1  # initialize them as not assigned
    maximum_average_spike_counts = [0] * population_exc

    for label in range(10):
        # get how many images exist in image_labels for current label.
        current_label_count = len(np.where(image_labels == label)[0])
        spike_counts_per_image = np.vstack(spike_counts_per_image)
        
        if current_label_count > 0:
            # calculate how many times each neuron fired for current label, during simulation.
            total_spike_counts = np.sum(spike_counts_per_image[image_labels == label], axis=0)
            # calculate average of total_spike_count_per_neuron by dividing it by image count with current label.
            average_spike_counts = total_spike_counts / current_label_count
            for neuron_idx in range(population_exc):
                if average_spike_counts[neuron_idx] > maximum_average_spike_counts[neuron_idx]:
                    maximum_average_spike_counts[neuron_idx] = average_spike_counts[neuron_idx]
                    assigned_labels[neuron_idx] = label
    np.save(f'{run_name}/assignments_from_training.npy', assigned_labels)

def _get_predictions_for_current_image(spike_counts_current_image: ty.List[int], assignments_from_training: np.ndarray) -> ty.List[int]:
    predictions = []
    for label in range(10):
        # get how many neurons are assigned to current label.
        assignment_indices = np.where(assignments_from_training == label)[0]
        assignment_count = len(assignment_indices)

        if assignment_count > 0:
            # Calculate total spike count for current label (sum of each neuron's spike count which belongs to same label).
            total_spike_count = np.sum(spike_counts_current_image[assignment_indices])
            # Calculate average spike count for current label.
            average_spike_count = total_spike_count / assignment_count
            predictions.append(average_spike_count)
        else:
            # Handle the case where no neurons are assigned to the current label.
            predictions.append(0)

    # Sort predictions in descending order of average spike counts
    predictions = np.argsort(predictions)[::-1]

    return list(predictions)

def get_predictions(spike_counts_per_image: ty.List[ty.List[int]], run_name: str) -> ty.List[ty.List[int]]:
    assignments_from_training = np.load(f'{run_name}/assignments_from_training.npy')
    test_image_count = len(spike_counts_per_image)
    predictions_per_image = []
    for image_idx in range(test_image_count):
        predictions_current_image = _get_predictions_for_current_image(spike_counts_per_image[image_idx], assignments_from_training)
        predictions_per_image.append(predictions_current_image)

    return predictions_per_image

def calculate_accuracy(predictions_per_image: ty.List[ty.List[int]], test_image_labels: np.ndarray) -> float:
    predictions_per_image_array = np.array(predictions_per_image)
    test_image_count = len(test_image_labels)
    
    # Assuming predictions_per_image has the top prediction as the first index
    top_predictions = predictions_per_image_array[:, 0]
    difference = top_predictions - test_image_labels
    correct = len(np.where(difference == 0)[0])
    incorrect = test_image_count - correct
    
    accuracy = (correct / test_image_count) * 100
    print(f"Accuracy: {accuracy:.2f}%, Correct: {correct}, Incorrect: {incorrect}")
    return accuracy