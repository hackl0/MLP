"""Prédiction seule : aucun entraînement ni appel externe."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import hashlib
import importlib.metadata
import json
import re
import unicodedata

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

MAX_CARACTERES = 2000


def harmoniser_texte(texte):
    texte = unicodedata.normalize("NFKC", texte).casefold()
    texte = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " ", texte)
    texte = re.sub(r"(?:https?://|www\.)\S+", " ", texte)
    texte = texte.replace("’", "'").replace("‘", "'")
    texte = re.sub(r"(?<!\w)#(?=\w)", "", texte)
    # Garder les lettres/chiffres et quelques signes porteurs de sens métier.
    texte = "".join(
        c if c.isalnum() or c in " '+#./-" else " " for c in texte
    )
    # Tirets et apostrophes : uniquement à l'intérieur d'un mot.
    texte = re.sub(r"(?<!\w)[-'’]+|[-'’]+(?!\w)", " ", texte)
    # + et # après un nombre ou seuls sont décoratifs ; après une lettre,
    # ils restent utiles pour les langages C++, C# ou F#.
    texte = re.sub(r"(?<![a-z+#])[+#]+", " ", texte)
    # Points isolés ou finaux et slashs isolés : séparateurs décoratifs.
    texte = re.sub(r"\.(?!\w)|(?<!\w)/(?!\w)", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


@dataclass(frozen=True)
class Proposition:
    category: str
    domaine: str
    score: float


class Predicteur:
    def __init__(self, dossier: Path | None = None):
        self.dossier = dossier or Path(__file__).resolve().parent / "models"
        self.metadata = json.loads((self.dossier / "metadata.json").read_text(encoding="utf-8"))
        # Vérifier les fichiers avant chargement et éviter un mélange d'expériences.
        for nom, empreinte in self.metadata["sha256"].items():
            if hashlib.sha256((self.dossier / nom).read_bytes()).hexdigest() != empreinte:
                raise RuntimeError(f"Le fichier {nom} ne correspond pas au modèle livré.")
        version = importlib.metadata.version("scikit-learn")
        if version != self.metadata["versions"]["scikit-learn"]:
            raise RuntimeError("La version de scikit-learn doit correspondre au requirements.txt livré.")
        self.modele = joblib.load(self.dossier / "metiers_meilleur_mlp.joblib")
        self.vectoriseur = joblib.load(self.dossier / "metiers_vectoriseur_tfidf.joblib")
        self.encodeur = joblib.load(self.dossier / "metiers_encodeur_categories.joblib")
        self.libelles = json.loads((self.dossier / "job_categories_20.json").read_text(encoding="utf-8-sig"))
        if self.modele.n_features_in_ != len(self.vectoriseur.vocabulary_):
            raise RuntimeError("Le MLP et le vocabulaire TF-IDF sont incompatibles.")
        if not np.array_equal(self.modele.classes_, np.arange(len(self.encodeur.classes_))):
            raise RuntimeError("Le modèle et l'encodeur des catégories sont incompatibles.")
        if self.encodeur.classes_.tolist() != self.metadata["categories"]:
            raise RuntimeError("La liste des catégories ne correspond pas aux artefacts.")
        self.domaines = [self.libelles[cat] for cat in self.encodeur.classes_]
        self._verrou = RLock()

    def predire(self, texte: str, nombre_choix: int = 3) -> list[Proposition]:
        if not isinstance(texte, str):
            raise ValueError("Saisis un intitulé de métier ou une courte description.")
        if len(texte) > MAX_CARACTERES:
            raise ValueError(f"Le texte doit contenir au maximum {MAX_CARACTERES} caractères.")
        if not isinstance(nombre_choix, int) or nombre_choix < 1:
            raise ValueError("Le nombre de propositions doit être positif.")
        texte_nettoye = harmoniser_texte(texte)
        if sum(c.isalpha() for c in texte_nettoye) < 2:
            raise ValueError("Ajoute un intitulé contenant au moins deux lettres.")
        vecteur = self.vectoriseur.transform([texte_nettoye])
        if vecteur.nnz == 0:
            raise ValueError("Aucun terme reconnu. Précise le métier ou ajoute quelques compétences.")
        with self._verrou, threadpool_limits(limits=2):
            scores = self.modele.predict_proba(vecteur)[0]
        positions = scores.argsort()[::-1][:min(nombre_choix, len(scores))]
        categories = self.encodeur.inverse_transform(self.modele.classes_[positions])
        return [Proposition(cat, self.libelles[cat], float(scores[pos])) for cat, pos in zip(categories, positions)]
