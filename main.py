# Importe l'index qui a été construit dans documents.py
# et s'assure que le LLM est configuré en important llm.py
from documents import index
import llm

def main():
    """
    Fonction principale pour exécuter la boucle de questions-réponses.
    """
    print("\nLe moteur de recherche (query engine) est prêt.")
    print("Posez vos questions sur le diabète. Tapez 'exit' pour quitter.")

    # Le query_engine est créé à partir de l'index.
    # Il utilisera automatiquement le LLM configuré dans Settings (via llm.py).
    query_engine = index.as_query_engine()

    # Boucle pour poser des questions en continu
    while True:
        try:
            # Demander une question à l'utilisateur
            question = input("\nVotre question : ")

            # Vérifier si l'utilisateur veut quitter
            if question.lower().strip() == 'exit':
                print("Au revoir !")
                break

            # Exécuter la requête
            print("Génération de la réponse...")
            response = query_engine.query(question)
            
            # Afficher la réponse
            print("\n--- Réponse ---")
            print(str(response))
            print("---------------")

        except KeyboardInterrupt:
            # Permet de quitter proprement avec Ctrl+C
            print("\nAu revoir !")
            break
        except Exception as e:
            print(f"\nUne erreur est survenue : {e}")
            break

if __name__ == "__main__":
    main()
