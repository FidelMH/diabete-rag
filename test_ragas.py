"""
Test et évaluation du système RAG avec RAGAS.

Ce script permet d'évaluer la qualité du système RAG en utilisant :
- Génération automatique d'un dataset de test
- Métriques RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall)
- LLM externe (Groq/OpenAI) pour une évaluation de qualité
"""

import os
from dotenv import load_dotenv
from documents import EmbedderRag

from llama_index.core import SimpleDirectoryReader
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

from ragas.testset import TestsetGenerator
from ragas.integrations.llama_index import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# Charger les variables d'environnement
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("La clé GROQ_API_KEY n'est pas définie dans le fichier .env")


class RAGEvaluator:
    """
    Classe pour évaluer un système RAG avec RAGAS.
    """

    def __init__(
        self,
        documents_path: str = "./documents",
        model_name: str = "openai/gpt-oss-20b",
        embed_model_name: str = "bge-m3",
        testset_size: int = 5
    ):
        """
        Initialise l'évaluateur RAG.

        Args:
            documents_path: Chemin vers les documents
            model_name: Modèle Groq pour l'évaluation
            embed_model_name: Modèle d'embedding Ollama
            testset_size: Nombre de questions de test à générer
        """
        self.documents_path = documents_path
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.testset_size = testset_size

        # Initialiser le LLM Groq pour l'évaluation (LangChain)
        self.llm = ChatGroq(
            model=self.model_name,
            api_key=API_KEY,
            temperature=0.3
        )

        # Initialiser le modèle d'embedding local (LangChain)
        self.embed_model = OllamaEmbeddings(
            model=self.embed_model_name,
        )

        print(f"✓ LLM initialisé: {self.model_name}")
        print(f"✓ Embedding initialisé: {self.embed_model_name}")

    def generate_testset(self):
        """
        Génère un dataset de test à partir des documents.

        Returns:
            Dataset de test RAGAS
        """
        print(f"\n📚 Chargement des documents depuis '{self.documents_path}'...")
        documents = SimpleDirectoryReader(self.documents_path).load_data()
        print(f"✓ {len(documents)} document(s) chargé(s)")

        print(f"\n🔧 Initialisation du générateur de testset...")
        # Wrapper les modèles LangChain pour RAGAS
        generator_llm = LangchainLLMWrapper(self.llm)
        generator_embeddings = LangchainEmbeddingsWrapper(self.embed_model)

        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )

        print(f"\n⚙️  Génération de {self.testset_size} questions de test...")
        print("   (Cette opération peut prendre quelques minutes)")
        testset = generator.generate_with_llamaindex_docs(
            documents,
            testset_size=self.testset_size,
        )

        print(f"✓ Testset généré avec succès!")
        return testset

    def evaluate_rag(self, testset):
        """
        Évalue le système RAG avec les métriques RAGAS.

        Args:
            testset: Dataset de test généré

        Returns:
            Résultats de l'évaluation
        """
        print(f"\n🏗️  Construction de l'index vectoriel...")
        embedder = EmbedderRag(
            model_name=self.embed_model_name,
            input_path=self.documents_path
        )
        index = embedder.build_or_load_index()

        print(f"\n🔍 Création du QueryEngine...")
        query_engine = index.as_query_engine()

        print(f"\n📊 Configuration des métriques d'évaluation...")
        evaluator_llm = LangchainLLMWrapper(self.llm)

        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
        ]

        print(f"\n🚀 Évaluation en cours...")
        print("   Métriques:")
        print("   - Faithfulness: Fidélité de la réponse aux documents")
        print("   - Answer Relevancy: Pertinence de la réponse")
        print("   - Context Precision: Précision du contexte récupéré")
        print("   - Context Recall: Rappel du contexte pertinent")

        ragas_dataset = testset.to_evaluation_dataset()

        result = evaluate(
            query_engine=query_engine,
            metrics=metrics,
            dataset=ragas_dataset,
        )

        return result

    def display_results(self, result):
        """
        Affiche les résultats de l'évaluation.

        Args:
            result: Résultats de l'évaluation RAGAS
        """
        print("\n" + "="*60)
        print("📈 RÉSULTATS DE L'ÉVALUATION")
        print("="*60)

        # Afficher les scores globaux
        if hasattr(result, 'scores'):
            print("\n🎯 Scores moyens:")
            for metric, score in result.scores.items():
                print(f"   {metric}: {score:.4f}")

        # Convertir en DataFrame pour une vue détaillée
        df = result.to_pandas()
        print("\n📋 Résultats détaillés:")
        print(df.to_string())

        # Sauvegarder les résultats
        output_file = "ragas_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Résultats sauvegardés dans '{output_file}'")

        return df


def main():
    """
    Fonction principale pour exécuter l'évaluation RAGAS.
    """
    print("="*60)
    print("🧪 ÉVALUATION RAG AVEC RAGAS")
    print("="*60)

    # Initialiser l'évaluateur
    evaluator = RAGEvaluator(
        documents_path="./documents",
        model_name="openai/gpt-oss-20b",  # Modèle Groq
        embed_model_name="bge-m3",  # Modèle d'embedding local
        testset_size=5  # Nombre de questions à générer
    )

    # Générer le testset
    testset = evaluator.generate_testset()

    # Évaluer le système RAG
    result = evaluator.evaluate_rag(testset)

    # Afficher les résultats
    df = evaluator.display_results(result)

    print("\n✅ Évaluation terminée avec succès!")


if __name__ == "__main__":
    main()
