from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
#import sklearn
import itertools
from scipy.sparse import hstack

from sklearn.metrics.pairwise import pairwise_distances
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
  #NB: ne fonctionne pas pour le moment: 'function' object has no attribute 'toarray'
  i = 0 
  for name, matrix in liste_repr:
    if name not in combi_names:
      continue
    elif i ==0:
      X = matrix
    else:
      if has_opti == True:
        X = concatenate_csc_matrices_by_columns 
      else:
        X = hstack((X, matrix))
    i+=1
  return X.toarray()

def vectorize_texts(liste_texts, mini = 3, maxi=4):
  V = CountVectorizer(analyzer="char", ngram_range=(mini, maxi))
  X = V.fit_transform(liste_texts)
  return X

def get_similarity(info_img, liste_repr, has_opti=False):#IN : liste de tuples (NOM_vectorisation ,liste_vecteurs)
  """
  info_img   : tuples PATH, liste_etiquettes
  liste_repr : tuples nom_repr, matrice
  """
  nom_repr = [x[0] for x in liste_repr]
  liste_combis = get_combination(nom_repr)
  print("combis", liste_combis)

  dic_out = {}
  for combi in liste_combis:
    dic_out.setdefault(combi, {})
    X = concat_matrix(combi, liste_repr, has_opti)
    for meth in ["braycurtis"]:
      matrix = pairwise_distances(X, Y=None, metric=meth)
      dic_out[combi][meth] = matrix
  #prevoir vecteurs vides
  #OUT : {combi : {meth:X}}
  return dic_out

if __name__=="__main__":
  texts = ["toto", "totototo", "titi"]
  print(vectorize_texts(texts))

  liste_reprs = []
  for m, M in [[3,4], [4, 5]]:
    X = vectorize_texts(texts, mini=m, maxi=M)
    liste_reprs.append([f"char_{m}-{M}", X])

  etiq = ["good", "bad", "ugly"]
  infos_imgs  = [[texts[i], etiq[i], []] for i in range(len(texts))]
  get_similarity(infos_imgs, liste_reprs)
