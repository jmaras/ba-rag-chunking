"""
flat_chunker.py – Flat-Chunking-Strategie: Sliding Window über komplette Dokumente.

Token-basiert, aggregiert alle Paragraphen eines Dokuments und erstellt
überlappende Chunks fester Größe (512 Tokens, 50 Overlap).
"""

from typing import List, Dict
import json
from pathlib import Path
from transformers import AutoTokenizer


class FlatChunker:
    """
    Sliding-Window-Chunking über den vollständigen Dokumenttext.

    Alle Paragraphen eines Dokuments werden zu einem String zusammengefügt,
    anschließend token-basiert in überlappende Chunks mit fester Größe aufgeteilt.
    Das char-basierte Splitting erhält Umlaute und Groß-/Kleinschreibung.
    """

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Args:
            chunk_size:    Max. Tokens pro Chunk
            chunk_overlap: Überlappende Tokens zwischen Chunks
            model_name:    Embedding-Modell (bestimmt den Tokenizer)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap muss kleiner als chunk_size sein.")

        print(f"   Lade Tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   Tokenizer geladen.")

    def _char_based_split(self, text: str, tokens: List[int]) -> List[str]:
        """
        Teilt den Text anhand von Token-Positionen auf, gibt aber den
        originalen Zeichenbereich zurück (erhält Sonderzeichen).
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False
        )
        offsets = encoding['offset_mapping']
        step_size = self.chunk_size - self.chunk_overlap
        chunk_texts = []

        for i in range(0, len(tokens), step_size):
            start_idx = i
            end_idx = min(i + self.chunk_size, len(tokens))

            if end_idx - start_idx < 50:
                continue

            if start_idx < len(offsets) and end_idx <= len(offsets):
                char_start = offsets[start_idx][0]
                char_end = offsets[end_idx - 1][1]
                chunk_texts.append(text[char_start:char_end])

        return chunk_texts

    def chunk_documents(self, parsed_docs: List[Dict]) -> List[Dict]:
        """
        Erstellt Flat-Chunks aus einer Liste geparster Dokumente.

        Strategie:
          1. Alle Paragraphen eines Dokuments zu einem String zusammenfügen.
          2. Gesamttext tokenisieren.
          3. Sliding Window mit chunk_size und chunk_overlap anwenden.
          4. Textstücke char-basiert aus dem Original extrahieren.

        Args:
            parsed_docs: Ausgabe von DocxParser.parse_all_documents()

        Returns:
            Liste von Chunk-Dictionaries mit Text und Metadaten
        """
        all_chunks = []
        chunk_id = 0

        for doc in parsed_docs:
            doc_filename = doc['metadata']['filename']
            paragraphs = doc['paragraphs']

            print(f"  Verarbeite {doc_filename}: {len(paragraphs)} Paragraphen")

            all_text = ' '.join([p['text'] for p in paragraphs if p['text'].strip()])
            if not all_text.strip():
                print(f"    Kein Text in {doc_filename} gefunden.")
                continue

            tokens = self.tokenizer.encode(all_text, add_special_tokens=False)
            step_size = self.chunk_size - self.chunk_overlap
            num_chunks_expected = max(1, (len(tokens) - self.chunk_overlap) // step_size)

            print(f"    Tokens: {len(tokens):,}  –  erwartet ~{num_chunks_expected} Chunks")

            chunk_texts = self._char_based_split(all_text, tokens)

            for chunk_text in chunk_texts:
                chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)

                # Kapitel/Abschnitt per Heuristik bestimmen (erste 5 Wörter)
                chapter = None
                section = None
                first_words = chunk_text.split()[:5]
                if first_words:
                    search_text = ' '.join(first_words)
                    for para in paragraphs:
                        if search_text in para['text']:
                            chapter = para.get('chapter')
                            section = para.get('section')
                            break

                all_chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        'doc_filename': doc_filename,
                        'chapter': chapter,
                        'section': section,
                        'chunking_strategy': 'flat',
                        'token_count': len(chunk_tokens),
                        'is_split': False
                    }
                })
                chunk_id += 1

        print(f"\n  {len(all_chunks)} Flat-Chunks erstellt.")
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
        return {
            'num_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'num_split_chunks': 0
        }