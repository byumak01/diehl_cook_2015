def increase_spiking_rates(image, current_max_rate):
    new_maximum_rate = current_max_rate + 32
    return (image * new_maximum_rate) / current_max_rate