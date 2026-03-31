"""
hierarchical_chunker.py – Hierarchisches Chunking, das Kapitel- und Abschnittsgrenzen respektiert.

Token-basiert. Chunks werden auf Section-Ebene erstellt und nur bei Überschreitung
von max_chunk_size aufgeteilt (kein Overlap – Grenzen bleiben erhalten).
"""

from typing import List, Dict
import json
from pathlib import Path
from transformers import AutoTokenizer


class HierarchicalChunker:
    """
    Hierarchisches Chunking auf Section-Ebene.

    Jede Section wird als eigenständiger Chunk behandelt. Ist sie größer als
    max_chunk_size, wird sie ohne Overlap aufgeteilt. Kapitel ohne Sections
    werden auf Kapitel-Ebene gechunkt.
    """

    def __init__(self,
                 max_chunk_size: int = 512,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Args:
            max_chunk_size: Max. Tokens pro Chunk
            model_name:     Embedding-Modell (bestimmt den Tokenizer)
        """
        self.max_chunk_size = max_chunk_size

        print(f"   Lade Tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   Tokenizer geladen.")

    def _split_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """
        Teilt Text token-basiert auf (kein Overlap).
        Gibt den originalen Zeichenbereich zurück, um Sonderzeichen zu erhalten.
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False
        )
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            end_idx = min(i + max_tokens, len(tokens))
            if end_idx - i < 20:
                continue
            char_start = offsets[i][0]
            char_end = offsets[end_idx - 1][1]
            chunks.append(text[char_start:char_end])

        return chunks

    def chunk_documents(self, parsed_docs: List[Dict]) -> List[Dict]:
        """
        Erstellt Hierarchical-Chunks aus einer Liste geparster Dokumente.

        Strategie:
          - Section mit Paragraphen ≤ max_chunk_size → ein Chunk pro Section.
          - Section > max_chunk_size → mehrere Chunks ohne Overlap.
          - Kapitel ohne Sections → Chunking auf Kapitel-Ebene.

        Args:
            parsed_docs: Ausgabe von DocxParser.parse_all_documents()

        Returns:
            Liste von Chunk-Dictionaries mit Text und Metadaten
        """
        all_chunks = []
        chunk_id = 0

        for doc in parsed_docs:
            doc_filename = doc['metadata']['filename']
            chapters = doc['chapters']

            print(f"  Verarbeite {doc_filename}: {len(chapters)} Kapitel")

            for chapter in chapters:
                chapter_title = chapter['title']
                sections = chapter.get('sections', [])

                if sections:
                    for section in sections:
                        section_title = section['title']
                        para_ids = section.get('paragraphs', [])

                        section_text_parts = []
                        for para_id in para_ids:
                            para = next((p for p in doc['paragraphs'] if p['id'] == para_id), None)
                            if para:
                                section_text_parts.append(para['text'])

                        if not section_text_parts:
                            continue

                        section_text = ' '.join(section_text_parts)
                        token_count = len(self.tokenizer.encode(section_text, add_special_tokens=False))

                        if token_count <= self.max_chunk_size:
                            all_chunks.append({
                                'id': chunk_id,
                                'text': section_text,
                                'metadata': {
                                    'doc_filename': doc_filename,
                                    'chapter': chapter_title,
                                    'section': section_title,
                                    'chunking_strategy': 'hierarchical',
                                    'hierarchy_level': 'section',
                                    'token_count': token_count,
                                    'is_split': False
                                }
                            })
                            chunk_id += 1

                        else:
                            section_chunks = self._split_text_by_tokens(section_text, self.max_chunk_size)
                            for split_idx, chunk_text in enumerate(section_chunks):
                                chunk_token_count = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
                                all_chunks.append({
                                    'id': chunk_id,
                                    'text': chunk_text,
                                    'metadata': {
                                        'doc_filename': doc_filename,
                                        'chapter': chapter_title,
                                        'section': section_title,
                                        'chunking_strategy': 'hierarchical',
                                        'hierarchy_level': 'section_split',
                                        'token_count': chunk_token_count,
                                        'is_split': True,
                                        'split_index': split_idx
                                    }
                                })
                                chunk_id += 1

                else:
                    # Kapitel ohne Sections
                    chapter_text_parts = [
                        p['text'] for p in doc['paragraphs']
                        if p.get('chapter') == chapter_title
                    ]
                    if not chapter_text_parts:
                        continue

                    chapter_text = ' '.join(chapter_text_parts)
                    chapter_chunks = self._split_text_by_tokens(chapter_text, self.max_chunk_size)

                    for chunk_text in chapter_chunks:
                        chunk_token_count = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
                        all_chunks.append({
                            'id': chunk_id,
                            'text': chunk_text,
                            'metadata': {
                                'doc_filename': doc_filename,
                                'chapter': chapter_title,
                                'section': None,
                                'chunking_strategy': 'hierarchical',
                                'hierarchy_level': 'chapter',
                                'token_count': chunk_token_count,
                                'is_split': len(chapter_chunks) > 1
                            }
                        })
                        chunk_id += 1

        print(f"\n  {len(all_chunks)} Hierarchical-Chunks erstellt.")
        return all_chunks

    def save_chunks(self, chunks: List[Dict], output_file: str):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"  {len(chunks)} Chunks gespeichert: {output_path}")

    def get_statistics(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {}
        token_counts = [c['metadata']['token_count'] for c in chunks]
        level_counts = {}
        for chunk in chunks:
            level = chunk['metadata']['hierarchy_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        return {
            'num_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'hierarchy_levels': level_counts,
            'num_split_chunks': sum(1 for c in chunks if c['metadata']['is_split'])
        }