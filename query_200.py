# from visual import Visualiser
import os
from models_200 import returnVisualFeatures, Visualiser
import pickle
import numpy as np
from annoy import AnnoyIndex
from image_viewer import  mv3
from preprocess_words import get_query_vector2
from metric import ang_avg

## Replace with folder you trained over, so that similar images can be retrieved
train_folder = "/Users/gsp/Downloads"


#### INPUT FILE NAMES #### 
query_image = "/Users/gsp/Downloads/images/MEN-Tees_Tanks-id_00000390-13_1_front.jpg"
#query_image = "/Users/gsp/Documents/photus/meee.jpeg"
#query_image = "/Users/gsp/Downloads/1516264345931.jpeg"
query_text = "floral" ## Make sure empty code handled

learnt_data_space = "models/vector_search_200.pkl"
training_dict_files = 'models/images_list_200.pkl'
distance_mode = 'angular'     # 'angular' is suitable for cosine similarity. Can also try "euclidean", "manhattan", "hamming", or "dot"
vector_length = 2148 ## 2048 + 100
text_embeddings_file = 'models/text_embedding_200.pkl'
text_weight =  1000

shape_labels = "/Users/gsp/Downloads/labels/shape/shape_anno_all.txt"
fabric_texture_labels = "/Users/gsp/Downloads/labels/texture/fabric_ann.txt"
pattern_texture_labels = "/Users/gsp/Downloads/labels/texture/pattern_ann.txt"
ourVisualiser = Visualiser()
ourVisualiser.load()


approx_nn_model = AnnoyIndex(vector_length, distance_mode)
approx_nn_model.load(learnt_data_space)

with open(text_embeddings_file, 'rb') as file:
    embeddings_model = pickle.load(file)

visual_features = np.array(returnVisualFeatures(ourVisualiser, query_image))
text_query_v_unw = np.array(get_query_vector2(query_text, embeddings_model))
text_query_vector = text_weight * np.array(get_query_vector2(query_text, embeddings_model))


## To be used for evaluating relevant nearest neighbours
concatenated_array = np.concatenate((visual_features, text_query_vector), axis=0)

## To be used for evaluating cosine similarity
query_vector_unweighted =  np.concatenate((visual_features, text_query_v_unw), axis=0)


with open(training_dict_files, 'rb') as file:
    images_list = pickle.load(file)

nearest_indices = approx_nn_model.get_nns_by_vector(concatenated_array, n=6)
print(nearest_indices)

# Retrieve the nearest words based on the indices
nearest_images = [os.path.join(train_folder, images_list[index]) for index in nearest_indices]

nearest_image_vectors = [approx_nn_model.get_item_vector(key) for key in nearest_indices]
avg_angle_score = ang_avg(query_vector_unweighted, nearest_image_vectors)
print("Final score efficiency = ", avg_angle_score)
print("Len of retrieved iamges = " , len(nearest_images))
print("Images = ", nearest_images)
mv3(nearest_images)
