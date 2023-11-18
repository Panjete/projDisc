import tensorflow as tf
import pandas as pd
import numpy as np
from color import *

# data = {
#     'Red': [255, 0, 128, 72],
#     'Green': [0, 128, 255, 150],
#     'Blue': [128, 255, 0, 200]
# }

color_dictionary = {
    0: 'Red',
    1: 'Green',
    2: 'Blue',
    3: 'Yellow',
    4: 'Orange',
    5: 'Pink',
    6: 'Purple',
    7: 'Brown',
    8: 'Grey',
    9: 'Black',
    10: 'White'
}
# Create a DataFrame
# df = pd.DataFrame(data)

import os

color_dict = {}


def list_files_in_folder(folder_path):
    i=0
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Print each file in the folder
        for file in files:
            if i%100 == 0:
                print(i)
            
            i+=1 
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                color_dict[file] = color_list(file_path)
                
    except OSError as e:
        print(f"Error reading the folder: {e}")

# Example usage:
folder_path = "images"
list_files_in_folder(folder_path)

file_name = "color.json"

import json
# Write the dictionary to the JSON file
with open(file_name, 'w') as json_file:
    json.dump(color_dict, json_file, indent=2)
# model = tf.keras.models.load_model('colormodel_trained_89.h5') #very important
# model.summary()

# train_predictions = model.predict(df)


# predicted_encoded_train_labels = np.argmax(train_predictions, axis=1)
# print(predicted_encoded_train_labels)

# for p in predicted_encoded_train_labels:
#     print(color_dictionary[p])
