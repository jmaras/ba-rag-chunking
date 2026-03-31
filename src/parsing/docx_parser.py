"""
docx_parser.py – Parser für Word-Dokumente mit 2-Ebenen-Hierarchie (Kapitel/Abschnitte).
"""

from docx import Document
from typing import Dict, List
from pathlib import Path
import json


class DocxParser:
    """Parst Word-Dokumente und extrahiert eine 2-Ebenen-Hierarchie (Kapitel + Abschnitte)."""

    def __init__(self):
        self.style_to_level = {
            'Heading 1': 1,
            'Heading 2': 2,
            'Überschrift 1': 1,
            'Überschrift 2': 2
        }

    def parse_document(self, filepath: Path) -> Dict:
        """
        Parst ein Word-Dokument und gibt die Dokumentstruktur zurück.

        Heading 1 → Kapitel, Heading 2 → Abschnitt, alles andere → Paragraph.
        Heading 3 und tiefer werden als normaler Text behandelt.

        Args:
            filepath: Pfad zum Word-Dokument

        Returns:
            Dict mit 'metadata', 'chapters' und 'paragraphs'
        """
        doc = Document(str(filepath))

        result = {
            'metadata': self._extract_metadata(filepath),
            'chapters': [],
            'paragraphs': []
        }

        current_chapter = None
        current_section = None
        para_id = 0

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            level = self.style_to_level.get(para.style.name)

            if level == 1:
                current_chapter = {'title': text, 'level': 1, 'sections': []}
                result['chapters'].append(current_chapter)
                current_section = None

            elif level == 2:
                if current_chapter:
                    current_section = {'title': text, 'level': 2, 'paragraphs': []}
                    current_chapter['sections'].append(current_section)

            else:
                para_obj = {
                    'id': para_id,
                    'text': text,
                    'chapter': current_chapter['title'] if current_chapter else None,
                    'section': current_section['title'] if current_section else None,
                    'style': para.style.name,
                    'word_count': len(text.split())
                }
                result['paragraphs'].append(para_obj)

                if current_section:
                    current_section['paragraphs'].append(para_id)

                para_id += 1

        return result

    def _extract_metadata(self, filepath: Path) -> Dict:
        return {
            'filename': filepath.stem,
            'filepath': str(filepath),
            'topic': 'XML-Validierung in GEFEG.FX'
        }

    def parse_all_documents(self, doc_paths: List[Path], output_file: Path):
        """
        Parst alle Dokumente und speichert das Ergebnis als JSON.

        Args:
            doc_paths:   Liste von Pfaden zu Word-Dokumenten
            output_file: Zielpfad für die JSON-Ausgabe

        Returns:
            Liste aller geparsten Dokumente
        """
        all_docs = []

        for path in doc_paths:
            print(f"Parsing: {path.name}")
            parsed = self.parse_document(path)
            all_docs.append(parsed)

            num_chapters = len(parsed['chapters'])
            num_sections = sum(len(ch['sections']) for ch in parsed['chapters'])
            num_paragraphs = len(parsed['paragraphs'])
            print(f"  {num_chapters} Kapitel, {num_sections} Abschnitte, {num_paragraphs} Paragraphen")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        total_paras = sum(len(d['paragraphs']) for d in all_docs)
        print(f"\n{len(all_docs)} Dokumente geparst, {total_paras} Paragraphen gesamt.")

        return all_docs

    def get_statistics(self, parsed_doc: Dict) -> Dict:
        stats = {
            'filename': parsed_doc['metadata']['filename'],
            'num_chapters': len(parsed_doc['chapters']),
            'num_sections': sum(len(ch['sections']) for ch in parsed_doc['chapters']),
            'num_paragraphs': len(parsed_doc['paragraphs']),
            'avg_paragraphs_per_section': 0
        }
        if stats['num_sections'] > 0:
            stats['avg_paragraphs_per_section'] = round(
                stats['num_paragraphs'] / stats['num_sections'], 1
            )
        return stats