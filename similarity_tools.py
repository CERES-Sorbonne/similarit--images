from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_texts(liste_texts):
  mini, maxi = 3, 4
  V = CountVectorizer(analyzer="char", ngram_range=(mini, maxi))
  X = V.fit_transform(liste_texts)
  return X

def get_similarity(lsite_vecteurs):#IN : liste de tuples (NOM_vectorisation ,liste_vecteurs)
  #prevoir vecturs vides
  # make combi (concat)
  # similarity matrix
  pass

if __name__=="__main__":
  texts = ["toto", "totototo", "titi"]
  print(vectorize_texts(texts))
