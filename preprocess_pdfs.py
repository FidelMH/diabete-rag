"""
Script de prétraitement et d'inspection des PDFs.

Ce script permet d'analyser les documents PDF avant de les indexer,
d'afficher des statistiques, et de vérifier la qualité du nettoyage.
"""

import argparse
from llama_index.core import SimpleDirectoryReader
from text_cleaner import TextCleaner, get_document_stats


def display_document_preview(doc, max_chars: int = 500):
    """
    Affiche un aperçu d'un document.

    Args:
        doc: Document à afficher
        max_chars: Nombre maximum de caractères à afficher
    """
    preview = doc.text[:max_chars]
    if len(doc.text) > max_chars:
        preview += "..."

    print(f"\n{'='*70}")
    print(f"Document: {doc.metadata.get('file_name', 'Sans nom')}")
    print(f"{'='*70}")
    print(preview)
    print(f"{'='*70}")
    print(f"Longueur totale: {len(doc.text)} caractères")
    print(f"Nombre de mots: {len(doc.text.split())} mots")


def analyze_documents(documents_path: str, preview: bool = False):
    """
    Analyse les documents d'un dossier.

    Args:
        documents_path: Chemin vers les documents
        preview: Si True, affiche un aperçu des documents
    """
    print(f"\n📂 Analyse des documents dans '{documents_path}'...")
    print("=" * 70)

    # Charger les documents
    documents = SimpleDirectoryReader(documents_path).load_data()
    print(f"\n✓ {len(documents)} document(s) chargé(s)")

    # Statistiques avant nettoyage
    print("\n📊 STATISTIQUES AVANT NETTOYAGE")
    print("-" * 70)
    stats_before = get_document_stats(documents)
    for key, value in stats_before.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Nettoyer les documents
    print("\n🧹 Nettoyage des documents en cours...")
    cleaned_documents = TextCleaner.clean_documents(
        documents,
        remove_urls=True,
        normalize_medical=False
    )
    print("✓ Nettoyage terminé!")

    # Statistiques après nettoyage
    print("\n📊 STATISTIQUES APRÈS NETTOYAGE")
    print("-" * 70)
    stats_after = get_document_stats(cleaned_documents)
    for key, value in stats_after.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Calcul de la réduction
    if stats_before['total_characters'] > 0:
        reduction_chars = ((stats_before['total_characters'] - stats_after['total_characters']) /
                         stats_before['total_characters'] * 100)
        reduction_words = ((stats_before['total_words'] - stats_after['total_words']) /
                         stats_before['total_words'] * 100)

        print("\n📉 RÉDUCTION")
        print("-" * 70)
        print(f"  Caractères: {reduction_chars:.1f}%")
        print(f"  Mots: {reduction_words:.1f}%")

    # Afficher aperçu si demandé
    if preview:
        print("\n\n👁️  APERÇUS DES DOCUMENTS")
        print("=" * 70)

        for i, (original, cleaned) in enumerate(zip(documents, cleaned_documents), 1):
            print(f"\n\n📄 DOCUMENT {i}/{len(documents)}")

            print("\n--- ORIGINAL ---")
            display_document_preview(original, max_chars=300)

            print("\n--- NETTOYÉ ---")
            display_document_preview(cleaned, max_chars=300)

            if i < len(documents):
                input("\n[Appuyez sur Entrée pour voir le document suivant...]")

    # Analyse des problèmes potentiels
    print("\n\n🔍 ANALYSE DES PROBLÈMES POTENTIELS")
    print("=" * 70)

    problems = []
    for i, doc in enumerate(documents, 1):
        # Vérifier les documents trop courts
        if len(doc.text) < 100:
            problems.append(f"Document {i}: Texte trop court ({len(doc.text)} caractères)")

        # Vérifier les lignes répétitives (possible header/footer)
        lines = doc.text.split('\n')
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        repetitive = [line for line, count in line_counts.items() if count > 5]
        if repetitive:
            problems.append(f"Document {i}: {len(repetitive)} ligne(s) répétitive(s) détectée(s)")

        # Vérifier la présence d'URLs
        if 'http' in doc.text.lower():
            url_count = doc.text.lower().count('http')
            problems.append(f"Document {i}: {url_count} URL(s) détectée(s)")

    if problems:
        for problem in problems:
            print(f"  ⚠️  {problem}")
    else:
        print("  ✓ Aucun problème majeur détecté!")

    # Recommandations
    print("\n\n💡 RECOMMANDATIONS")
    print("=" * 70)

    recommendations = []

    # Recommander nettoyage si réduction significative
    if stats_before['total_characters'] > 0:
        reduction = ((stats_before['total_characters'] - stats_after['total_characters']) /
                    stats_before['total_characters'] * 100)
        if reduction > 10:
            recommendations.append(
                f"Le nettoyage réduit le texte de {reduction:.1f}%. "
                "Il est recommandé d'activer clean_text=True dans EmbedderRag."
            )

    # Recommander suppression d'URLs si présentes
    has_urls = any('http' in doc.text.lower() for doc in documents)
    if has_urls:
        recommendations.append(
            "Des URLs ont été détectées. Il est recommandé d'activer remove_urls=True."
        )

    # Recommander normalisation médicale
    medical_terms = ['DT1', 'DT2', 'HbA1c', 'IMC']
    has_medical_abbr = any(
        any(term in doc.text for term in medical_terms)
        for doc in documents
    )
    if has_medical_abbr:
        recommendations.append(
            "Des abréviations médicales ont été détectées. "
            "Vous pouvez activer normalize_medical=True pour les normaliser."
        )

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  ✓ Les documents semblent en bon état!")

    print("\n" + "=" * 70)
    print("✅ Analyse terminée!\n")


def main():
    """
    Fonction principale du script de prétraitement.
    """
    parser = argparse.ArgumentParser(
        description="Analyse et prétraitement des documents PDF pour le RAG"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./documents",
        help="Chemin vers le dossier contenant les documents (défaut: ./documents)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Afficher un aperçu des documents avant/après nettoyage"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔬 ANALYSE ET PRÉTRAITEMENT DES DOCUMENTS PDF")
    print("=" * 70)

    analyze_documents(args.path, preview=args.preview)


if __name__ == "__main__":
    main()
