## For evaluating performance of the retrieval system
import numpy as np


def cos_sim(array1, array2):
    # array1 = np.array(arr1)
    # array2 = np.array(arr2)
    dot_product = np.dot(array1, array2)
    norm_array1 = np.linalg.norm(array1)
    norm_array2 = np.linalg.norm(array2)
    return dot_product / (norm_array1 * norm_array2) # = cosine_similarity 

def ang_avg(arr1, arrs2):
    avg_angle = 0.0
    for arr in arrs2:
        a = cos_sim(arr, arr1)
        avg_angle += a
        print("Angle found between query and retrieved : ", a)
    return avg_angle/len(arrs2)
