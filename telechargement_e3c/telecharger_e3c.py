from urllib.request import urlretrieve
from urllib.parse import quote  #pour gérer les URL avec des espaces (control character)
from pathlib import Path, PurePath 
import json
import subprocess

URL_BASE = "http://quandjepasselebac.education.fr/e3c/"
URL_SOURCE = "http://quandjepasselebac.education.fr/e3c/index_14-05-2020.json"
INDEX_JSON = "index-passelebac.json"
DEBUG = False

def menu(liste_noms):    
    print('-' * 30 + 'Faites votre choix ("x" pour sortir) ' + '-' * 30)
    for k, nom_matiere in enumerate(liste_noms):
        print(f'{nom_matiere}, choix {k}')
    return input('Saisir votre choix :')

#téléchargement si besoin du fichier d'index json
if not Path(INDEX_JSON).exists():
   _, __  = urlretrieve(URL_SOURCE, INDEX_JSON)


with open(INDEX_JSON) as f:
    data = json.load(f)
    les_matieres_generales = list(data['items'][0]['items'])
    les_noms_des_matieres_generales = [matiere['name'] for matiere in les_matieres_generales]
    les_index_matieres_genererales = list(map(str, range(len(les_noms_des_matieres_generales))))
    les_specialites = list(data['items'][0]['items'][1]['items'])
    les_noms_des_specialites = [matiere['name'] for matiere in les_specialites]
    les_index_specialites = list(map(str, range(len(les_noms_des_specialites))))
    if DEBUG:
        print(les_noms_des_matieres_generales)
        print(les_noms_des_specialites)
    choix_general = menu(les_noms_des_matieres_generales)
    while  choix_general in les_index_matieres_genererales:
        matiere_choisie = les_matieres_generales[int(choix_general)]
        if matiere_choisie['name'] == "Enseignements de spécialité":
            print("spe")
            choix_specialite = menu(les_noms_des_specialites)
            matiere_choisie = les_specialites[int(choix_specialite)]
        nom_matiere = matiere_choisie['name']
        if not Path(nom_matiere).exists():
            subprocess.call(['mkdir', nom_matiere])
        repertoire  = PurePath(nom_matiere)
        liste_sujets_e3c2 = list(matiere_choisie['items'][0]['items'])
        if DEBUG:
            print(f'Liste des sujets E3C2 en {nom_matiere} :')
            print(liste_sujets_e3c2)
        for sujet in liste_sujets_e3c2:
            print(f"Téléchargement du sujet {sujet['name']} de la matière {nom_matiere}")
            if not Path(repertoire / sujet['name']).exists():
                if DEBUG:
                    print(URL_BASE + sujet['path'])
                _, __  = urlretrieve(URL_BASE + quote(sujet['path']), repertoire / sujet['name'])
        choix_general = menu(les_noms_des_matieres_generales)


    
