"""Interface Streamlit du classifieur de métiers."""
import streamlit as st

from inference import MAX_CARACTERES, Predicteur


st.set_page_config(page_title="Quel domaine pour ce métier ?", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Chargement du modèle…")
def charger_predicteur():
    return Predicteur()


st.caption("PROJET MACHINE LEARNING · JOSEPH LAROCHE")
st.title("Quel domaine pour ce métier ?")
st.write("Écris un intitulé de poste ou une courte bio. Mon réseau de neurones propose le domaine professionnel qui lui correspond.")

try:
    predicteur = charger_predicteur()
except Exception:
    st.error("Le modèle ne peut pas être chargé. Vérifie les fichiers et les versions indiqués dans le README du dépôt.")
    st.stop()

with st.form("prediction"):
    intitule = st.text_area(
        "Intitulé du métier ou courte bio",
        value="Développeur Python backend",
        placeholder="Exemple : Responsable recrutement et formation",
        max_chars=MAX_CARACTERES,
        height=130,
        help="Un intitulé précis ou quelques compétences donnent plus de contexte au modèle.",
    )
    soumettre = st.form_submit_button("Trouver le domaine", type="primary")

if soumettre:
    try:
        propositions = predicteur.predire(intitule)
    except ValueError as erreur:
        st.warning(str(erreur))
    else:
        st.subheader("Domaine proposé")
        st.success(propositions[0].domaine)
        st.write("Les trois premiers choix du modèle :")
        st.dataframe(
            [{"Domaine": p.domaine, "Score du modèle (%)": round(100 * p.score, 2)} for p in propositions],
            hide_index=True,
        )
        st.caption("Ces scores ne sont pas calibrés : un score élevé ne garantit pas une réponse correcte.")

st.info("Ce premier modèle choisit parmi 13 domaines. Il peut proposer un domaine inadapté si le métier appartient à une catégorie qu’il n’a pas apprise.")

with st.expander("Voir les 13 domaines disponibles"):
    for domaine in predicteur.domaines:
        st.markdown(f"- {domaine}")

with st.expander("Comment fonctionne mon IA ?"):
    st.write("Le texte est nettoyé, transformé en nombres avec TF-IDF, puis classé par un réseau MLP de 128 et 64 neurones. Le modèle est déjà entraîné : il ne réapprend pas avec les textes saisis ici.")
    st.write("Les exemples d’apprentissage viennent de descriptions LinkedIn classées initialement avec DeepSeek. Les catégories automatiques n’ont pas toutes été vérifiées à la main.")
    gauche, droite = st.columns(2)
    gauche.metric("Bonnes réponses sur le test", f"{100 * predicteur.metadata['accuracy_test']:.2f} %")
    droite.metric("F1 macro sur le test", f"{predicteur.metadata['f1_macro_test']:.4f}")
    st.caption("Résultats sur 1 198 textes réservés au test, dans les 13 catégories retenues. Ils ne garantissent pas les performances sur tous les métiers.")

st.caption("Aucun appel à DeepSeek pour cette prédiction. Le code de l’application ne sauvegarde pas les textes saisis.")
