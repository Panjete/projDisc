"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm
from fashion200.query_200 import Nearest_images
import pickle

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

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  
    # compute test query features
  ourVisualiser = Visualiser()

  for t in tqdm(test_queries):
    img = testset.get_img(t['source_img_id'])
    text = t['mod']['str']
    #print("FIRST t = ", t)
    #print("IMGS = ", imgs)
    #print("mods = ", mods)
    img_component = returnVisualfromPIL(ourVisualiser, img) 
    text_component = get_query_vector2(text, embeddings_model).tolist()
    #print("IMG COMPONENT = ", len(img_component[0]))
    #print("TEXT COMPONENT = ", len(text_component[0]))
    f = [np.array(img_component+text_component)]
    #print("F SHAPE = ", f[0].shape)
    all_queries += [f]
  print("LEN ALL QUERIES = ", len(all_queries))
  all_queries = np.concatenate(all_queries)
  print("ALL QUERIES SHAPE = ", all_queries.shape)
  all_target_captions = [t['target_caption'] for t in test_queries]

  # compute all image features
  for i in tqdm(range(len(testset.imgs))):
    img = testset.get_img(i)
    img = np.array(returnVisualfromPIL(ourVisualiser, img))
    #imgs = model.extract_img_feature(imgs).data.cpu().numpy()
    all_imgs += [img]
  all_imgs = np.concatenate(all_imgs)
  all_captions = [img['captions'][0] for img in testset.imgs]


  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
    nn_result.append(np.argsort(-sims[0, :])[:110])

  # compute recalls
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    out += [('recall_top' + str(k) + '_correct_composition', r)]
  return out



dataset = Fashion200k("/Users/gsp/Downloads/f200", 'test')
test(dataset)