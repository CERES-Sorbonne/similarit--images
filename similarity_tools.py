from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
#import sklearn
import itertools
from scipy.sparse import hstack
from scipy.sparse import vstack
import glob
import numpy
import json
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from main import load_ground_truth
import os
import pickle
from eval_tools import evaluate_sim_methods

def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csc_matrix((new_data, new_indices, new_ind_ptr))


def get_combination(names):
  """ Returns all combination of names"""
  liste_possible = []
  for L in range(0, len(names)+1):
    for subset in itertools.combinations(names, L):
      liste_possible.append(tuple(sorted(list(subset))))
  liste_possible = [x for x in liste_possible if len(x)>0]
  return sorted(liste_possible)

def concat_matrix(combi_names, liste_repr, has_opti = False):
  #TODO: has_opti est-il utile ?
  #concat lente si grosse matrice ? cf : https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices/33259578#33259578
  i = 0 
  for name, matrix in liste_repr:
    #TODO: un assert sur le nombre de lignes ?
    if name not in combi_names:
      continue
    elif i ==0:
      if type(matrix) is numpy.ndarray:
        X = matrix.tolist()
      else:
        #TODO: check this (dimensionnality reduction for OCR)
        X = [x[:1000] for x in matrix.toarray()]
    else:
      if type(matrix) is numpy.ndarray:
        X = [X[i]+matrix[i].tolist() for i in range(len(X))]
        #X = hstack((X, matrix.tolist()))
      else:
        X = hstack((X, matrix))
    i+=1
  #return X.toarray()
  return X
  #np array ?return X.tolist()

def vectorize_texts(liste_texts, mini = 3, maxi=4):
  #TODO: ajouter filtre pour réduire la dimensionnalité
  #min doc frequency, N plus fréquents (ou combinaison)
  V = CountVectorizer(analyzer="char", ngram_range=(mini, maxi))
  X = V.fit_transform(liste_texts)
  return X

def get_similarity(info_img, liste_repr, has_opti=False):#IN : liste de tuples (NOM_vectorisation ,liste_vecteurs)
  """
  IN: 2 listes
  info_img   : tuples PATH, liste_etiquettes
  liste_repr : tuples nom_repr, matrice
  OUT: un dictionnaire qui associe
  des combinaisons de représentations en entrée
    à une matrice de similarité selon des mesures (définies en dur)
      
  """
  nom_repr = [x[0] for x in liste_repr]
  liste_combis = get_combination(nom_repr)
  print("Combinaisons", liste_combis, "\n")

  dic_out = {}
  for combi in liste_combis:
    dic_out.setdefault(combi, {})
    X = concat_matrix(combi, liste_repr, has_opti)
    try:
      print("Shape",combi, len(X[0]))
    except:
      print("Shape",combi, X[0].shape)
    for meth in ["braycurtis", "cosine", "euclidean"]:#, "jaccard", "dice"]:
      #Need Boolean to int pour jacard et Dice
      #ajouter combinaisons (moyenne...)
      print("  ",meth)
      if meth =="cosine":
        matrix = cosine_distances(X)
      else:
        matrix = pairwise_distances(X, Y=None, metric=meth)
      dic_out[combi][meth] = matrix
  #prevoir vecteurs vides
  #OUT : {combi : {meth:X}}
  print("\n Similarity matrices computed\n")
  return dic_out

def test_with_GT(json_GT, force= False):
  data_GT = load_ground_truth(json_GT)
  print("\#Objets:", len(data_GT["image_list"]))
  liste_repr = [[name, matrice] for name, matrice, typ in data_GT["matrices"]][1:3]
  json_GT_sim = f"{json_GT}_sim_clip.pkl"
  if os.path.exists(json_GT_sim) and force==False:
    print(f"Working with existing {json_GT_sim}")
    with open(json_GT_sim, "rb") as f:
      data_sim = pickle.load(f)
  else:
    print(f"Re-computing {json_GT_sim}")
    data_sim = get_similarity(data_GT["image_list"], liste_repr)
    with open(json_GT_sim, "wb") as w:
      pickle.dump(data_sim, w)
  #TODO: régler dimensionnalité COuntVec
  #TODO: avant de passer à l'échelle, sauvegarder ?

  dic_results = evaluate_rank(data_GT["image_list"],data_sim)
  path_results = "dic_results_clip.pkl"
  with open(path_results, "wb") as w:
    pickle.dump(dic_results, w)
  print(f"Similarity output written : {path_results}")

  out_name_results= f"{path_results}_evaluation.json"
  evaluate_sim_methods(dic_results, out_name_results)

def evaluate_rank(info_img, data_sim, NB_most_simil = 20):
  #IN : {combi : {meth:X}}
  dic_results = {}
  for combi, dic_meth in data_sim.items():
    dic_results.setdefault(combi, {})
    for meth, sim_matrix in dic_meth.items():
      dic_results[combi].setdefault(meth, [])
      for i, info in enumerate(info_img):
        #NB: on enlève les nan, miam
        selected = [[sim_matrix[i][j], j] for j in range(len(sim_matrix[i]))]
        most_sim=sorted([[s,r] for s,r in selected if numpy.isnan(s)==False])
        res= []
        for sim, ID in most_sim[:NB_most_simil]:
          res.append([sim, info_img[ID]])
        dic_results[combi][meth].append(res)
  return dic_results

if __name__=="__main__":
  json_GT = f"ground_truth.json"
  if os.path.exists(json_GT)==True:
    test_with_GT(json_GT, force = True)
  else:
    print(f"{json_GT} missing, exiting ....")
  exit()
  # Old testing stuff :
  texts = ["toto", "totototo", "titi"]
  print(vectorize_texts(texts))

  liste_reprs = []
  for m, M in [[3,4], [4, 5]]:
    X = vectorize_texts(texts, mini=m, maxi=M)
    liste_reprs.append([f"char_{m}-{M}", X])

  etiq = ["good", "bad", "ugly"]
  infos_imgs  = [[texts[i], etiq[i], []] for i in range(len(texts))]
  for config, dic_metric in get_similarity(infos_imgs, liste_reprs).items():
    print(config, dic_metric)
