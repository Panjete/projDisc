import os
from visual import returnVisualFeatures, Visualiser
from classifier import returnOneHot, returnTextWords, Classifier
import json
import pickle
import numpy as np
from annoy import AnnoyIndex
from preprocess_words import get_query_vector2




#### INPUT FILES ####

pwd = "/Users/gsp/Desktop/SemVII/COL764/projbackup/"

folder_path = "/Users/gsp/Downloads/images"
caption_file = pwd + "inception_df_color/color_appended_caption.json"
test_embedding_file = pwd + 'models/colored_text_embedding.pkl'
shape_labels = "/Users/gsp/Downloads/labels/shape/shape_anno_all.txt"
fabric_texture_labels = "/Users/gsp/Downloads/labels/texture/fabric_ann.txt"
pattern_texture_labels = "/Users/gsp/Downloads/labels/texture/pattern_ann.txt"


#### OUTPUT FILES ####
generic_text = "sleeveless print block pure color"
learnt_data_space = pwd + "models/colored_vector_search.pkl"
training_dict_files = pwd + 'models/colored_images_list.pkl'

distance_mode = 'angular'    # 'angular' is suitable for cosine similarity. Can also try "euclidean", "manhattan", "hamming", or "dot"
ourClassifier = Classifier(shape_labels_file=shape_labels, fabric_texture_file=fabric_texture_labels, pattern_file=pattern_texture_labels)
ourClassifier.load()
ourVisualiser = Visualiser()
ourVisualiser.load()


# List all files in the folder -> FILENAMES, not FILEPATHS
file_list = os.listdir(folder_path)

folder_path_models = os.path.join(os.getcwd(), "models")
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
    labels = np.array(get_query_vector2(" ".join(returnTextWords(ourClassifier, filepath)), embeddings_model))
    one_hot_encoded_vectors = np.array(returnOneHot(ourClassifier, filepath))

    concatenated_array = np.concatenate((tensor1, text_vector, labels, one_hot_encoded_vectors), axis=0)
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