# Classer un métier avec un réseau MLP

Application Streamlit du projet de Joseph Laroche. Saisir un intitulé ou une courte bio pour afficher le domaine prédit et les trois premiers choix du modèle.

Le dépôt contient le modèle déjà entraîné. Aucun entraînement, dataset LinkedIn, appel API ou clé DeepSeek n’est nécessaire pour lancer l’application.

## Lancer en local

Depuis la racine de ce dépôt, avec Python **3.12** :

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Ouvrir ensuite l’adresse locale affichée par Streamlit. Sous Linux/macOS, remplacer `.venv\Scripts\python.exe` par `.venv/bin/python`.

## Mettre en ligne sur Streamlit Community Cloud

1. Envoyer les fichiers de ce dossier sur le dépôt GitHub `hackl0/MLP`, en conservant les dossiers `models` et `.streamlit`.
2. Se connecter à [Streamlit Community Cloud](https://share.streamlit.io/) avec son compte GitHub.
3. Choisir **Create app**, puis le dépôt `hackl0/MLP` et la branche contenant ces fichiers.
4. Mettre **`app.py`** dans **Main file path**.
5. Dans **Advanced settings**, sélectionner **Python 3.12**. Le fichier `.python-version` sert de repère local ; sélectionner explicitement la version dans Streamlit Cloud.
6. Lancer le déploiement. Aucune variable secrète n’est nécessaire.
7. Copier l’adresse `https://…streamlit.app` pour partager la démonstration.

Les fichiers `requirements.txt` et `app.py` doivent être à la racine du dépôt GitHub. L’application utilise des chemins relatifs à ses fichiers et fonctionne indépendamment du dossier du cours.

Instructions officielles : [déployer une application](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) et [installer les dépendances](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Fichiers nécessaires

```text
app.py                         Interface web
inference.py                   Nettoyage et prédiction
requirements.txt               Versions des bibliothèques
.streamlit/config.toml          Apparence et paramètres Streamlit
.python-version                Version Python de référence
models/
  metiers_meilleur_mlp.joblib
  metiers_vectoriseur_tfidf.joblib
  metiers_encodeur_categories.joblib
  job_categories_20.json
  metadata.json                Catégories, résultats et empreintes des fichiers
```

Les modèles sont des fichiers ordinaires du dépôt. Le plus gros fait environ 13 Mo : aucun téléchargement externe ni Git LFS n’est nécessaire pour cette version. Ne pas les exclure avec `.gitignore`.

## Fonctionnement et limites

- Le nettoyage reprend exactement la fonction du notebook du projet.
- TF-IDF utilise 6 080 variables. Le MLP possède deux couches cachées de 128 puis 64 neurones.
- Le modèle connaît **13 domaines**, parmi les 20 catégories prévues après regroupement. Les catégories rares et `unknown`/`other` ont été mises à part pendant la préparation.
- Sur les 1 198 textes de test : **77,46 % d’accuracy**, **0,7211 de F1 macro**.
- Les scores affichés pour un nouvel intitulé ne sont pas calibrés. Un score élevé ne garantit pas la justesse de la réponse.
- Un métier hors des 13 domaines peut être classé à tort dans l’un d’eux. Le modèle ne dispose pas d’un rejet fiable.
- Les textes sans aucun terme reconnu donnent un message invitant à reformuler.
- Les étiquettes initiales proviennent de DeepSeek et ne sont pas toutes validées à la main.
- Le code de l’application ne sauvegarde pas les saisies et ne lance aucun apprentissage en ligne.

## Vérifier l’installation

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Compatibilité des fichiers sauvegardés

Utiliser les versions de `requirements.txt`, notamment **scikit-learn 1.9.1**, pour charger le modèle dans les conditions de l’entraînement. Voir la [documentation scikit-learn sur la sauvegarde des modèles](https://scikit-learn.org/stable/model_persistence.html).

Le chargement vérifie les empreintes des fichiers dans `models/metadata.json`. Si un nouveau modèle est publié, remplacer ensemble le modèle, le vectoriseur, l’encodeur, les libellés et les métadonnées. Ne pas remplacer uniquement un des fichiers. Charger uniquement les fichiers joblib de confiance livrés avec le projet.
