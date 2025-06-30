
def get_norm_dict():

    synapse_counts = [313600, 202500, 142884, 82944, 32400]

    # Compute x values for the equation: x * 400 / synapse_count = 0.10
    norm_values = [(0.10 * count) / 400 for count in synapse_counts]

    # Pair each synapse count with its computed x
    results = list(zip(synapse_counts, norm_values))
    norm_dict = dict(results)

    return norm_dict
