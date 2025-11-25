"""
Test et évaluation du système RAG avec RAGAS.

Ce script permet d'évaluer la qualité du système RAG en utilisant :
- Génération automatique d'un dataset de test
- Métriques RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall)
- LLM externe (Groq/OpenAI) pour une évaluation de qualité
"""

import os
import sys

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from documents import EmbedderRag
from text_cleaner import TextCleaner
from embedding_manager import EmbeddingManager
from llm import LlmManager

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from ragas.testset import TestsetGenerator
from ragas.integrations.llama_index import evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# Charger les variables d'environnement
load_dotenv()


class RAGEvaluator:
    """
    Classe pour évaluer un système RAG avec RAGAS.
    """

    def __init__(
        self,
        documents_path: str = "./documents",
        testset_size: int = 5,
        convert_to_nodes: bool = True,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        """
        Initialise l'évaluateur RAG.

        Args:
            documents_path: Chemin vers les documents
            testset_size: Nombre de questions de test à générer
            convert_to_nodes: Si True, convertit les documents en chunks avant RAGAS
            chunk_size: Taille des chunks pour SentenceSplitter (match EmbedderRag)
            chunk_overlap: Chevauchement entre chunks (match EmbedderRag)
        """
        self.documents_path = documents_path
        self.testset_size = testset_size
        self.convert_to_nodes = convert_to_nodes
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialiser le LLM et l'Embedding Model via les managers
        # Cela configure les Settings globaux de LlamaIndex
        self.llm_manager = LlmManager()
        self.embedding_manager = EmbeddingManager()

        self.llm = Settings.llm
        self.embed_model = Settings.embed_model

    def generate_testset(self):
        """
        Génère un dataset de test à partir des documents.

        Returns:
            Dataset de test RAGAS
        """
        print(f"\n📚 Chargement des documents depuis '{self.documents_path}'...")
        documents = SimpleDirectoryReader(self.documents_path).load_data()
        print(f"✓ {len(documents)} document(s) chargé(s)")

        # Nettoyer les documents pour améliorer la qualité du testset
        print("\n🧹 Nettoyage des documents...")
        documents = TextCleaner.clean_documents(
            documents,
            remove_urls=True,
            normalize_medical=False
        )
        print("✓ Documents nettoyés")

        # Convertir les documents en chunks plus petits si activé (pour éviter l'erreur headlines)
        if self.convert_to_nodes:
            print("\n🔧 Conversion des documents en chunks...")
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            nodes = splitter.get_nodes_from_documents(documents)

            # Convertir les nodes en Documents avec métadonnées headlines
            chunked_documents = []
            for node in nodes:
                # Créer un nouveau Document à partir du node
                chunk_doc = Document(
                    text=node.get_content(),
                    metadata={
                        **node.metadata,
                        'headlines': []  # Garantir la présence de headlines
                    }
                )
                chunked_documents.append(chunk_doc)

            print(f"✓ {len(chunked_documents)} chunks créés avec métadonnées headlines")
            input_data = chunked_documents
        else:
            print("⚠️  Mode documents (pas de pre-chunking)")
            input_data = documents

        print(f"\n🔧 Initialisation du générateur de testset...")
        # # Wrapper les modèles LangChain pour RAGAS
        # generator_llm = LangchainLLMWrapper(self.llm)
        # generator_embeddings = LangchainEmbeddingsWrapper(self.embed_model)

        generator = TestsetGenerator.from_llama_index(
            llm=self.llm,
            embedding_model=self.embed_model,
        )

        print(f"\n⚙️  Génération de {self.testset_size} questions de test...")
        print(f"   Mode: {'Nodes' if self.convert_to_nodes else 'Documents'}")
        print("   (Cette opération peut prendre quelques minutes)")

        testset = generator.generate_with_llamaindex_docs(
            input_data,
            testset_size=self.testset_size,
            transforms=[],  # Désactive HeadlinesExtractor et HeadlineSplitter
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
            input_path=self.documents_path
        )
        index = embedder.build_or_load_index()

        print(f"\n🔍 Création du QueryEngine...")
        query_engine = index.as_query_engine(llm=self.llm)

        print(f"\n📊 Configuration des métriques d'évaluation...")
        evaluator_llm = LlamaIndexLLMWrapper(self.llm)

        # Configuration du modèle d'embedding pour RAGAS
        # RAGAS utilise Langchain pour les embeddings, nous devons donc wrapper notre modèle
        ragas_embeddings = LangchainEmbeddingsWrapper(
            LangchainOllamaEmbeddings(model=self.embed_model_name)
        )

        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=ragas_embeddings),
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

        df = result.to_pandas()

        # Afficher les scores globaux (moyenne de chaque métrique)
        if not df.empty:
            print("\n🎯 Scores moyens:")
            # Calculate the mean for numeric columns only
            for metric in df.select_dtypes(include='number').columns:
                mean_score = df[metric].mean(skipna=True)
                print(f"   {metric}: {mean_score:.4f}")

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
        # model_name="openai/gpt-oss-20b",  # Modèle Groq
        # embed_model_name="bge-m3",  # Modèle d'embedding local
        testset_size=2,  # Nombre de questions à générer
        convert_to_nodes=True  # Pre-chunking pour éviter l'erreur headlines
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
