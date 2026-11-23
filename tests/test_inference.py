"""Contrôles de la prédiction et des cas de saisie non exploitables."""
import unittest

from inference import Predicteur, harmoniser_texte


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predicteur = Predicteur()

    def test_termes_techniques_preserves(self):
        self.assertEqual(harmoniser_texte("C++ C# .NET Node.js C/C++"), "c++ c# .net node.js c/c++")

    def test_suppression_bruit(self):
        self.assertEqual(harmoniser_texte("'--Ingénieur | #Cloud 🚀 https://exemple.fr a@exemple.fr"), "ingénieur cloud")

    def test_developpeur_et_domaines(self):
        resultat = self.predicteur.predire("Développeur Python backend")
        self.assertEqual(resultat[0].category, "software")
        self.assertEqual(len(resultat), 3)
        self.assertGreaterEqual(resultat[0].score, resultat[1].score)
        self.assertTrue(all(0 <= r.score <= 1 for r in resultat))
        self.assertEqual(len(self.predicteur.domaines), 13)

    def test_variantes_de_decoration(self):
        self.assertEqual(self.predicteur.predire("#Développeur | Python backend 🚀"), self.predicteur.predire("développeur python backend"))

    def test_saisies_inexploitables(self):
        for texte in ["", "---", "a", "https://exemple.fr", "zzzzqwertyunknownword", "x" * 2001]:
            with self.subTest(texte=texte[:30]):
                with self.assertRaises(ValueError):
                    self.predicteur.predire(texte)


if __name__ == "__main__":
    unittest.main()
