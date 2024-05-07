import os

import numpy as np
from PIL import Image


def get_data():
    # get image paths
    base_data_path = 'data'
    image_paths = []

    for data_type in os.listdir(base_data_path):
        cur_path = f'{base_data_path}/{data_type}'
        for image_name in os.listdir(cur_path):
            image_paths.append(f'{cur_path}/{image_name}')

    print(len(image_paths))

    data = []
    labels = []
    for path in image_paths:
        img = Image.open(path).resize((224, 224, 3))
        img = np.array(img)
        label = 0 if 'without_mask' in path else 1
        data.append(img)
        labels.append(label)
	
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

