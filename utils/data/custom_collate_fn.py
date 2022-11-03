def collate_variable_images(original_batch):
    filtered_data = []
    filtered_target = []
    for item in original_batch:
        filtered_data.append(item[0])
        filtered_target.append(item[1])
    return filtered_data, filtered_target
