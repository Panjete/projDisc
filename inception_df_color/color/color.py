from colorthief import ColorThief
import matplotlib.pyplot as plt
from webcolors import rgb_to_name
import tensorflow as tf
import pandas as pd
import numpy as np

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

def create_rgb_dataframe(rgb_list):
    # Ensure that the input list is not empty
    if not rgb_list:
        print("Input list is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Create a DataFrame using the provided RGB list
    df = pd.DataFrame(rgb_list, columns=['Red', 'Green', 'Blue'])
    
    return df

# def closest_colour(requested_colour):
#     min_colours = {}
#     for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
#         r_c, g_c, b_c = webcolors.hex_to_rgb(key)
#         rd = (r_c - requested_colour[0]) ** 2
#         gd = (g_c - requested_colour[1]) ** 2
#         bd = (b_c - requested_colour[2]) ** 2
#         min_colours[(rd + gd + bd)] = name
#     return min_colours[min(min_colours.keys())]

# def get_colour_name(requested_colour):
#     try:
#         closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
#     except ValueError:
#         closest_name = closest_colour(requested_colour)
#         actual_name = None
#     return actual_name, closest_name


# color_thief = ColorThief('4.png')
# palette = color_thief.get_palette(color_count=2, quality=1)
# df = create_rgb_dataframe(palette)
# model = tf.keras.models.load_model('colormodel_trained_89.h5') #very important
# # model.summary()

# train_predictions = model.predict(df)


# predicted_encoded_train_labels = np.argmax(train_predictions, axis=1)
# print(predicted_encoded_train_labels)

# for p in predicted_encoded_train_labels:
#     print(color_dictionary[p])
model = tf.keras.models.load_model('colormodel_trained.h5') #very important

def color_list(image):
    ans_list = set()
    color_thief = ColorThief(image)
    palette = color_thief.get_palette(color_count=3, quality=20)
    df = create_rgb_dataframe(palette)
    # model.summary()

    train_predictions = model.predict(df, verbose=0)


    predicted_encoded_train_labels = np.argmax(train_predictions, axis=1)
    # print(predicted_encoded_train_labels)

    for p in predicted_encoded_train_labels:
        ans_list.add(color_dictionary[p])

    return list(ans_list)
