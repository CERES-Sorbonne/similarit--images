import os
import pickle
import json
import numpy

import scipy.sparse
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

INPUT_IMAGES_FOLDER = r"D:\Alie\Documents\Projets\AnalysesImages\outputs\vectors_gj\Steaks\IMG_Edouard"
INPUT_VECTORS_FOLDER = r"D:\Alie\Documents\Projets\AnalysesImages\outputs\vectors_gj"
VECTOR_LIST = ["dic_ocr.json", "dic_vec_no_pca.pkl", "dic_vec_pca_10.pkl", "dic_vec_pca_50.pkl", "dic_vec_ahashs.pkl"]


def load_vectors(vector_list):
    """
    Prend une liste de fichiers de vecteurs en entrée, load les pickle et retourne un dictionnaire du type:
    nom_type_vecteur -> {
        image -> vecteur
    }
    """
    vectors_dic = {}
    for vector in vector_list:
        if vector.endswith('json'):
            with open(os.path.join(INPUT_VECTORS_FOLDER, vector), 'r', encoding="utf-8") as f:
                # on a besoin d'une table de correspondance image -> ahash, pcke l'ocr est en format ahash -> ocr
                with open(os.path.join(INPUT_VECTORS_FOLDER, 'dic_ahashs.pkl'), 'rb') as f2:
                    image_to_ahash = pickle.load(f2)
                vec = _create_ocr_vectors(json.load(f), image_to_ahash)
        else:
            with open(os.path.join(INPUT_VECTORS_FOLDER, vector), 'rb') as f:
                vec = pickle.load(f)
        vectors_dic[vector.split('.')[0].split('dic_')[1]] = vec
    return vectors_dic
                

def _vectorize_texts(liste_texts):
  mini, maxi = 3, 4
  V = CountVectorizer(analyzer="char", ngram_range=(mini, maxi))
  X = V.fit_transform(liste_texts)
  return X


def _create_ocr_vectors(dic_ahashs_text, dic_image_ahashs):
    """
    Fonction utilitaire pour transformer un dico ahash -> text, et un dico image -> ahashs en dico image -> text_vector
    """
    dic_image_text = {}
    images_tuples = []
    for image, ahash2 in dic_image_ahashs.items():
        found = False
        for ahash, text in dic_ahashs_text.items():
            if ahash == ahash2:
                images_tuples.append((image, text))
                found = True
                break
        if not found:
            images_tuples.append((image, ''))
    text_matrix = _vectorize_texts([_tuple[1] for _tuple in images_tuples])
    for image, text_vector in zip(images_tuples, text_matrix):
        dic_image_text[image[0]] = text_vector
    return dic_image_text


def _init_matrices(vectors_dic):
    matrices = []
    for name, vector_dic in vectors_dic.items():
        first_key = list(vector_dic.keys())[0]
        first_element = vector_dic[first_key]
        if scipy.sparse.issparse(first_element):
            type = "sparse"
            obj = scipy.sparse.csr_matrix(first_element.shape, dtype=numpy.int64)
        else:
            type = "matrix"
            obj = []
        matrices.append({'type': type, 'name': name, 'obj': obj})
    return matrices


def create_ground_truth(vectors_dics):
    """
    Créé une liste de tuples de type (path, [etiquettes]), et une liste de noms de fichiers de matrices ('nom', 'nom_fichier', 'type')
    """
    ground_truth = {
        'image_list': [],
        'matrices': []
    }
    error_images = []
    # on initialise les matrices à persister
    matrices = _init_matrices(vectors_dics)

    for folder in os.listdir(INPUT_IMAGES_FOLDER):
        # on recompose les matrices en sélectionnant uniquement les vecteurs des images utilisées ici
        # et en respectant bien à chaque fois le même ordre
        for image in os.listdir(os.path.join(INPUT_IMAGES_FOLDER, folder)):
            # la clé qui a été utilisée pour mes dictionnaires de vecteurs n'a pas l'averagehash en début de nom
            dic_key = '_'.join(image.split('_')[1:])
            if dic_key == '':
                error_images.append(image)
                continue
            ground_truth['image_list'].append((
                os.path.join(folder, image),
                [folder, folder.split(' ')[0]],
            ))
            # on concatène les matrices différemment en fonction de si ce sont juste des vecteurs ou des sparses matrices
            for matrix in matrices:
                vectors_dic = vectors_dics[matrix['name']]
                if matrix['type'] == 'sparse':
                    matrix['obj'] = scipy.sparse.vstack((matrix['obj'], vectors_dic[dic_key]))
                else:
                    matrix['obj'].append(vectors_dic[dic_key])

    # on sauvegarde si il y a eu des erreurs
    with open('error_images.json', 'w') as f:
        json.dump(error_images, f)

    # on persiste les matrices
    for matrix in matrices:
        if matrix['type'] == 'sparse':
            file_name = f'{matrix["name"]}.npz'
            scipy.sparse.save_npz(file_name, matrix['obj'])
        else:
            file_name = f'{matrix["name"]}.npy'
            numpy.save(file_name, matrix['obj'])
        ground_truth['matrices'].append((matrix['name'], file_name, matrix['type']))
    return ground_truth


def persist_ground_truth():
    vectors_dic = load_vectors(VECTOR_LIST)
    gtruth = create_ground_truth(vectors_dic)
    with open('ground_truth.json', 'w') as f:
        json.dump(gtruth, f)


def create_similarity_matrices(input_vectors):
    """
    @input_vectors: un dictionnaire de type {'nom', <vecteur>}
    """
    pass


def combine_vectors(input_vectors):
    """
    Créé les toutes les combinaisons de vecteurs possibles.
    A voir si ça a vraiment un intérêt de créer 
    return: un dictionnaire de type {'combinaison de noms', <vecteur>}
    """
    pass


def evaluate():
    pass


def pipeline():
    pass


def load_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    for matrix in ground_truth['matrices']:
        if matrix[2] == 'sparse':
            matrix[1] = scipy.sparse.load_npz(matrix[1])
        else:
            matrix[1] = numpy.load(matrix[1])
    return ground_truth

if __name__ == "__main__":
    # persist_ground_truth()
    ground_truth = load_ground_truth('ground_truth.json')
    print("fini")