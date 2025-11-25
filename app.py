import streamlit as st
from documents import EmbedderRag
from llm import LlmManager

# Configuration de la page
st.set_page_config(
    page_title="Assistant Diabète",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design médical
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """
    Initialise le système RAG (LLM + Index).
    Cette fonction est mise en cache pour éviter de recharger le système à chaque interaction.
    """
    with st.spinner("Initialisation du système RAG..."):
        # Initialiser le LLM
        LlmManager()

        # Construire ou charger l'index
        embedder = EmbedderRag(input_path="./documents")
        index = embedder.build_or_load_index()

        # Créer le moteur de requête
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )

    return query_engine


def main():
    # Titre et description
    st.title("🩺 Assistant Médical - Diabète")
    st.markdown("""
    Posez vos questions sur le diabète et obtenez des réponses basées sur des documents médicaux.
    """)

    # Sidebar avec informations
    with st.sidebar:
        st.header("📋 Informations")
        st.markdown("""
        **Système RAG sur le Diabète**

        Ce système utilise :
        - LlamaIndex pour l'indexation
        - Ollama (bge-m3) pour les embeddings
        - Documents médicaux sur le diabète

        ---

        **Instructions :**
        1. Tapez votre question dans le champ ci-dessous
        2. Appuyez sur Entrée ou cliquez sur Envoyer
        3. Attendez la réponse basée sur les documents
        """)

        if st.button("🗑️ Effacer l'historique"):
            st.session_state.messages = []
            st.rerun()

    # Initialiser le système RAG
    try:
        query_engine = initialize_rag()
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du système : {e}")
        st.stop()

    # Initialiser l'historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie de la question
    if prompt := st.chat_input("Posez votre question sur le diabète..."):
        # Ajouter le message de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Afficher le message de l'utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)

        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("Génération de la réponse..."):
                try:
                    response = query_engine.query(prompt)
                    response_text = str(response)

                    # Afficher la réponse
                    st.markdown(response_text)

                    # Afficher les sources dans un expander
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        with st.expander("📚 Voir les sources"):
                            for idx, node in enumerate(response.source_nodes, 1):
                                st.markdown(f"**Source {idx}** (Score: {node.score:.3f})")
                                st.text(node.text[:300] + "..." if len(node.text) > 300 else node.text)
                                st.divider()

                    # Ajouter la réponse à l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

                except Exception as e:
                    error_msg = f"Une erreur est survenue : {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
