from documents import EmbedderRag
from llm import LlmManager

def main():
    """
    Fonction principale pour orchestrer le RAG et la boucle de questions-réponses.
    """
    try:
        # 1. Initialiser le LLM. 
        # Note: LlmManager applique le LLM aux Settings globaux lors de son initialisation.
        # J'utilise le modèle que vous aviez mis dans votre fichier llm.py.
        LlmManager()

        # 2. Construire ou charger l'index des documents.
        # J'ai remis le chemin vers le dossier, comme dans votre dernier fichier.
        embedder = EmbedderRag(input_path="./documents")
        index = embedder.build_or_load_index()

        # 3. Créer le moteur de requête avec vos paramètres optimisés.
        print("\nCréation du moteur de recherche (query engine)...")
        query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
        print("Le moteur de recherche est prêt.")
        
        # 4. Lancer la boucle interactive pour poser des questions.
        print("Posez vos questions sur le diabète. Tapez 'exit' pour quitter.")
        while True:
            question = input("\nVotre question : ")
            if question.lower().strip() == 'exit':
                print("Au revoir !")
                break

            if not question.strip():
                continue

            print("Génération de la réponse...")
            response = query_engine.query(question)
            
            print("\n--- Réponse ---")
            print(str(response))
            print("---------------")

    except KeyboardInterrupt:
        print("\nAu revoir !")
    except Exception as e:
        print(f"\nUne erreur critique est survenue : {e}")

if __name__ == "__main__":
    main()