import json

def  evaluate_one_object(res):
  dic_res = { 0: {i: {"VP":0, "FP":0, "FN":0} for i in range(1,len(res)+1)},
              1: {i: {"VP":0, "FP":0, "FN":0} for i in range(1,len(res)+1)}}
  etiq_ref = res[0][1][1]#le premier élément c'est l'image elle même
  for rang, similaire in enumerate(res[1:]):
    etiq_similaire = similaire[1][1]
    #cpt : c'est le rang dans la liste d'étiquettes
    for cpt, etiq in enumerate(etiq_similaire):
      if rang+1>1: 
        dic_res[cpt][rang+1]["VP"]+=dic_res[cpt][rang]["VP"]
        dic_res[cpt][rang+1]["FP"]+=dic_res[cpt][rang]["FP"]
      if etiq_ref[cpt] ==etiq:
        dic_res[cpt][rang+1]["VP"] += 1
      else:
        dic_res[cpt][rang+1]["FP"] += 1
  return dic_res

def get_FN(dic_res, res):
  etiq_ref = res[0][1][1]#le premier élément c'est l'image elle même
  #maintenant on gère les FN là où on s'est arrêté
  tot_VP = []#tous les VP de la liste pour calculer les FN
  for cpt in [0,1]:
    tot_VP.append(len([x for x in res[1:] if etiq_ref[cpt]==x[1][1][cpt]]))
  for cpt in [0,1]:
    for i in range(1,len(res)+1):
      dic_res[cpt][i]["FN"] = tot_VP[cpt]-dic_res[cpt][i]["VP"]
      if i>1:
        valeurs_cour = sum(list(dic_res[cpt][i].values()))
        valeurs_prec = sum(list(dic_res[cpt][i-1].values()))
        if valeurs_cour<=valeurs_prec:
          #on n'avait pas assez de résultats pour ce "rang"
          dic_res[cpt][i] = dic_res[cpt][i-1]
  return dic_res

def update(dic_res_glob, dic_res):#update dic_res_glob
  for cpt in [0,1]:
    for i in dic_res[cpt].keys():
      for cle in dic_res[cpt][i].keys():
        dic_res_glob[cpt].setdefault(i, {"VP":0, "FP":0, "FN":0})
        dic_res_glob[cpt][i][cle] += dic_res[cpt][i][cle]
  return dic_res_glob

def evaluate_tag_list(liste_res):
  dic_res_glob = { 0:{}, 1:{}}
  for res in liste_res:
    dic_res = evaluate_one_object(res)
    dic_res = get_FN(dic_res, res)
    dic_res_glob = update(dic_res_glob, dic_res)#update dic_res_glob
  return dic_res_glob

def compute_RPF(dic):
  R = dic["VP"] / (dic["VP"]+dic["FN"])
  P = dic["VP"] / (dic["VP"]+dic["FP"])
  F = (2*P*R)/(P+R)
  return R, P, F

def evaluate_sim_methods(sim_data, out_name= "results_RPF.json"):
  #ajouter évaluation seuil ?
  D = {}
  for config_name, dic_config in sim_data.items():
    print(config_name)
    config_name = str(config_name)
    D.setdefault(config_name, {})
    for meth_name, liste_res in dic_config.items():
      print(" ",meth_name)
      D[config_name].setdefault(meth_name, {})
      res = evaluate_tag_list(liste_res)
      for level, dic_level in res.items():
        print("  ",level)
        rangs = sorted(dic_level.keys())
        l_F = []
        all_res = []
        for rang in rangs:
          R, P, F = compute_RPF(dic_level[rang])
          all_res.append({"R":R, "P":P, "F":F})
          l_F.append(round(F, 4))
        print(" F-mes",l_F)
        D[config_name][meth_name][level] = all_res
  with open(out_name, "w") as w:
    w.write(json.dumps(D, indent =2))
  print(f"Results written in : {out_name}")

if __name__=="__main__":
  import pickle
  pickle_in = "dic_results_clip.pkl"
  print(f"Test évaluation sur {pickle_in}")
  with open(pickle_in, "rb") as f:
    data_sim = pickle.load(f)
  
  import os
  path1 = "results_eval"
  os.makedirs(path1, exist_ok = True)
  
  import re
  path2 = re.sub("/", "__", pickle_in)
  out_name = f"{path1}/{path2}.json"
  evaluate_sim_methods(data_sim, out_name )
