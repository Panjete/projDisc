import os
from .models_200 import returnVisualFeatures, Visualiser, get_image_paths_relative
import json
import pickle
import numpy as np
from annoy import AnnoyIndex
from preprocess_words import get_query_vector2

#### OUTPUT FILES ####
generic_text = "woman"
learnt_data_space = "/Users/gsp/Desktop/SemVII/COL764/projbackup/models/vector_search_200.pkl"
training_dict_files = '/Users/gsp/Desktop/SemVII/COL764/projbackup/models/images_list_200.pkl'


#### INPUT FILES ####
folder_path = "/Users/gsp/Downloads"
caption_file = "captions_200.json"
test_embedding_file = '/Users/gsp/Desktop/SemVII/COL764/projbackup/models/text_embedding_200.pkl'


distance_mode = 'angular'    # 'angular' is suitable for cosine similarity. Can also try "euclidean", "manhattan", "hamming", or "dot"
ourVisualiser = Visualiser()
ourVisualiser.load()


# List all files in the folder -> FILENAMES, not FILEPATHS
file_list = get_image_paths_relative()

if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
    os.makedirs(folder_path)

with open(caption_file, 'r') as f:
    captions = json.load(f)

with open(test_embedding_file, 'rb') as file:
    embeddings_model = pickle.load(file)

training_data = {}
kk = 0
for filename in file_list:
    filepath = os.path.join(folder_path, filename)
    tensor1 = np.array(returnVisualFeatures(ourVisualiser, filepath))
    if filename in captions:
        text_vector = np.array(get_query_vector2(captions[filename], embeddings_model))
    else:
        text_vector = np.array(get_query_vector2(generic_text, embeddings_model))

    concatenated_array = np.concatenate((tensor1, text_vector), axis=0)
    training_data[filename] = concatenated_array
    kk += 1
    if kk%100 == 0:
        print("Training K = ", kk)


vector_dim = len(next(iter(training_data.values())))
t = AnnoyIndex(vector_dim, distance_mode) 

for i, vector in enumerate(training_data.values()):
    t.add_item(i, vector)  # Add items with unique IDs

t.build(n_trees=10)
t.save(learnt_data_space)

images_list = list(training_data.keys())
with open(training_dict_files, 'wb') as file:
    pickle.dump(images_list, file)