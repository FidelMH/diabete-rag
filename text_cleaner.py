"""
Module de nettoyage et prétraitement du texte extrait des PDFs.

Ce module fournit des fonctions pour nettoyer et normaliser le texte
extrait des documents PDF, améliorant ainsi la qualité des embeddings
et des réponses du système RAG.
"""

import re
from typing import List
from llama_index.core.schema import Document


class TextCleaner:
    """
    Classe pour nettoyer et prétraiter le texte extrait des documents.
    """

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        Nettoie les espaces blancs excessifs et normalise les sauts de ligne.

        Args:
            text: Texte à nettoyer

        Returns:
            Texte nettoyé
        """
        # Remplacer plusieurs espaces par un seul
        text = re.sub(r' +', ' ', text)

        # Remplacer plusieurs sauts de ligne par maximum 2
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Supprimer les espaces en début et fin de ligne
        text = '\n'.join(line.strip() for line in text.split('\n'))

        return text.strip()

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """
        Supprime les numéros de page communs dans les PDFs.

        Args:
            text: Texte à nettoyer

        Returns:
            Texte sans numéros de page
        """
        # Supprimer les lignes contenant uniquement un ou plusieurs chiffres
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Supprimer les patterns comme "Page 1", "p. 23", etc.
        text = re.sub(r'\b[Pp]age\s+\d+\b', '', text)
        text = re.sub(r'\bp\.\s*\d+\b', '', text)

        return text

    @staticmethod
    def remove_headers_footers(text: str, min_repetition: int = 3) -> str:
        """
        Détecte et supprime les headers/footers répétitifs.

        Args:
            text: Texte à nettoyer
            min_repetition: Nombre minimum de répétitions pour considérer comme header/footer

        Returns:
            Texte sans headers/footers répétitifs
        """
        lines = text.split('\n')

        # Compter la fréquence de chaque ligne courte (probable header/footer)
        line_counts = {}
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line and len(cleaned_line) < 100:  # Headers/footers sont généralement courts
                line_counts[cleaned_line] = line_counts.get(cleaned_line, 0) + 1

        # Identifier les lignes répétitives
        repetitive_lines = {
            line for line, count in line_counts.items()
            if count >= min_repetition
        }

        # Filtrer les lignes répétitives
        filtered_lines = [
            line for line in lines
            if line.strip() not in repetitive_lines
        ]

        return '\n'.join(filtered_lines)

    @staticmethod
    def clean_special_characters(text: str) -> str:
        """
        Nettoie les caractères spéciaux problématiques tout en préservant la ponctuation utile.

        Args:
            text: Texte à nettoyer

        Returns:
            Texte nettoyé
        """
        # Remplacer les caractères unicode problématiques
        replacements = {
            '\u00a0': ' ',  # Espace insécable
            '\u2018': "'",  # Apostrophe courbe gauche
            '\u2019': "'",  # Apostrophe courbe droite
            '\u201c': '"',  # Guillemet courbe gauche
            '\u201d': '"',  # Guillemet courbe droit
            '\u2013': '-',  # Tiret moyen
            '\u2014': '-',  # Tiret long
            '\u2026': '...',  # Points de suspension
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Supprimer les caractères de contrôle sauf \n et \t
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text

    @staticmethod
    def fix_hyphenation(text: str) -> str:
        """
        Corrige les mots coupés par des tirets en fin de ligne (hyphenation).

        Args:
            text: Texte à corriger

        Returns:
            Texte avec mots recollés
        """
        # Coller les mots coupés (mot- \n mot devient mot-mot)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        return text

    @staticmethod
    def remove_urls_and_emails(text: str) -> str:
        """
        Supprime les URLs et adresses email du texte.

        Args:
            text: Texte à nettoyer

        Returns:
            Texte sans URLs ni emails
        """
        # Supprimer les URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Supprimer les emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        return text

    @staticmethod
    def normalize_medical_terms(text: str) -> str:
        """
        Normalise certains termes médicaux pour améliorer la cohérence.

        Args:
            text: Texte à normaliser

        Returns:
            Texte normalisé
        """
        # Normaliser les abréviations courantes du diabète
        replacements = {
            r'\bDT1\b': 'diabète de type 1',
            r'\bDT2\b': 'diabète de type 2',
            r'\bHbA1c\b': 'hémoglobine glyquée',
            r'\bIMC\b': 'indice de masse corporelle',
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    @staticmethod
    def clean_document(doc: Document,
                      remove_urls: bool = True,
                      normalize_medical: bool = False,
                      min_repetition: int = 3) -> Document:
        """
        Applique toutes les étapes de nettoyage à un document LlamaIndex.

        Args:
            doc: Document LlamaIndex à nettoyer
            remove_urls: Si True, supprime les URLs et emails
            normalize_medical: Si True, normalise les termes médicaux
            min_repetition: Nombre minimum de répétitions pour les headers/footers

        Returns:
            Document nettoyé
        """
        text = doc.text

        # Appliquer les nettoyages dans l'ordre
        text = TextCleaner.clean_special_characters(text)
        text = TextCleaner.fix_hyphenation(text)
        text = TextCleaner.remove_page_numbers(text)
        text = TextCleaner.remove_headers_footers(text, min_repetition)

        if remove_urls:
            text = TextCleaner.remove_urls_and_emails(text)

        if normalize_medical:
            text = TextCleaner.normalize_medical_terms(text)

        text = TextCleaner.clean_whitespace(text)

        # Créer un nouveau document avec le texte nettoyé
        cleaned_doc = Document(
            text=text,
            metadata={
                **doc.metadata,
                'cleaned': True,
                'headlines': []  # Ajouter metadata headlines vide pour éviter les erreurs RAGAS
            }
        )

        return cleaned_doc

    @staticmethod
    def clean_documents(documents: List[Document],
                       remove_urls: bool = True,
                       normalize_medical: bool = False,
                       min_repetition: int = 3) -> List[Document]:
        """
        Nettoie une liste de documents.

        Args:
            documents: Liste de documents à nettoyer
            remove_urls: Si True, supprime les URLs et emails
            normalize_medical: Si True, normalise les termes médicaux
            min_repetition: Nombre minimum de répétitions pour les headers/footers

        Returns:
            Liste de documents nettoyés
        """
        return [
            TextCleaner.clean_document(
                doc,
                remove_urls=remove_urls,
                normalize_medical=normalize_medical,
                min_repetition=min_repetition
            )
            for doc in documents
        ]


def get_document_stats(documents: List[Document]) -> dict:
    """
    Calcule des statistiques sur une liste de documents.

    Args:
        documents: Liste de documents

    Returns:
        Dictionnaire avec les statistiques
    """
    total_chars = sum(len(doc.text) for doc in documents)
    total_words = sum(len(doc.text.split()) for doc in documents)
    avg_chars_per_doc = total_chars / len(documents) if documents else 0
    avg_words_per_doc = total_words / len(documents) if documents else 0

    return {
        'num_documents': len(documents),
        'total_characters': total_chars,
        'total_words': total_words,
        'avg_chars_per_document': avg_chars_per_doc,
        'avg_words_per_document': avg_words_per_doc,
    }
