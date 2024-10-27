def check_update(curr_image_idx, update_interval):
    if curr_image_idx % update_interval == 0 and curr_image_idx != 0:
        return True
    else:
        return False