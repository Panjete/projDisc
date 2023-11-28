"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm
from fashion200.query_200 import Nearest_images, nearest_n_eval
import pickle
import time

from datasets import Fashion200k
from fashion200.models_200 import returnVisualfromPIL, Visualiser
from fashion200.preprocess_words import get_query_vector2
test_embedding_file = '/Users/gsp/Desktop/SemVII/COL764/projbackup/models/text_embedding_200.pkl'
with open(test_embedding_file, 'rb') as file:
    embeddings_model = pickle.load(file)


def test(testset):
  """Tests a model over the given testset."""
  #model.eval()
  test_queries = testset.get_test_queries()
  print("Test queries ;", test_queries[:5])
  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  
    # compute test query features
  ourVisualiser = Visualiser()
  jj = 0

  hamaare_saare_captions = []
  for t in tqdm(test_queries):
    img = testset.get_img(t['source_img_id'])
    text = t['mod']['str'].split()[-1]
    
    img_component = returnVisualfromPIL(ourVisualiser, img) 
    text_component = get_query_vector2(text, embeddings_model).tolist()
    top110 = nearest_n_eval(img_component, text_component)
    top110_paths = [path[21:] for path in top110]
    top110_indices = []
    for path in top110_paths:
      if path in testset.reversemap:
        top110_indices += [testset.reversemap[path]]
      else:
        top110_indices += [1234]
    top110_captions = [testset.imgs[idx]['captions'][0] for idx in top110_indices]
    hamaare_saare_captions += [top110_captions]
    jj +=1
    if jj %1000 == 0:
      #time.sleep(30)
      #print("sleeping because reached = ", jj)
      pass
  
  all_target_captions = [t['target_caption'] for t in test_queries]
  print("LEN ALL target captions = ", len(all_target_captions))
  jj = 0

  
  

  print("ALL TARGET CAPTIONS : \n :", all_target_captions[:5])
  # compute recalls
  out = []
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(hamaare_saare_captions):
      if all_target_captions[i] in nns[:k]:
        r += 1

    r /= len(hamaare_saare_captions)
    out += [('recall_top' + str(k) + '_correct_composition', r)]
    print('recall_top' , k , '_correct_composition', r)
    with open("logs.txt", 'a') as fi:
      fi.write('recall_top' + str(k) + '_correct_composition' + str(r) + "\n")
  return out



dataset = Fashion200k("/Users/gsp/Downloads/f200", 'test')
test(dataset)