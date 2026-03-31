"""
xml_validation_rag.py – RAG-System für GEFEG.FX-Schulungsmaterialien.

Unterstützt beide Chunking-Strategien (flat / hierarchical).
"""

from pathlib import Path
import json
from typing import List, Dict, Optional
import sys
import numpy as np
import pickle

script_dir = Path(__file__).resolve().parent
if script_dir.name == 'rag':
    project_root = script_dir.parent.parent
elif script_dir.name == 'src':
    project_root = script_dir.parent
else:
    project_root = script_dir

sys.path.insert(0, str(project_root / 'src'))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_DEPS = True
except ImportError as e:
    print(f"Fehlende Abhängigkeiten: {e}")
    print("  pip install faiss-cpu sentence-transformers transformers torch bitsandbytes")
    HAS_DEPS = False


class XMLValidationRAG:
    """
    RAG-System für GEFEG.FX-Schulungsmaterialien.

    Retrieval-Modi:
      flat          – Sliding-Window-Chunks, nach Ähnlichkeit sortiert
      hierarchical  – Sections-basierte Chunks, mit Chapter-Kohärenz-Boost
    """

    def __init__(self,
                 retrieval_mode: str = 'flat',
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 llm_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
                 index_dir: Optional[Path] = None):
        """
        Args:
            retrieval_mode:  'flat' oder 'hierarchical'
            embedding_model: Sentence-Transformer-Modell für Embeddings
            llm_model:       LLM für die Antwortgenerierung
            index_dir:       Verzeichnis für FAISS-Index-Dateien
        """
        if not HAS_DEPS:
            raise RuntimeError("Erforderliche Abhängigkeiten fehlen.")

        self.retrieval_mode = retrieval_mode
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.project_root = project_root
        self.index_dir = index_dir or (project_root / 'data' / 'faiss_index')

        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.index = None
        self.chunks = None

        # Boost-Faktor für Chapter-Kohärenz im hierarchischen Modus
        self.hierarchical_boost = 0.15

        print(f"RAG-System initialisiert (Modus: {self.retrieval_mode})")

    def load_embedding_model(self):
        if self.embedding_model is None:
            print(f"\nLade Embedding-Modell: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"  Embedding-Modell geladen.")

    def load_llm(self):
        """Lädt das LLM mit 4-bit-NF4-Quantisierung."""
        if self.llm_model is None:
            print(f"\nLade LLM: {self.llm_model_name}")

            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"  LLM geladen (GPU-Speicher: {mem:.2f} GB)")
            else:
                print(f"  LLM geladen (CPU)")

    def load_chunks(self, chunks_file: Optional[Path] = None, force_rebuild: bool = False):
        """
        Lädt Chunks und erstellt (oder lädt) den FAISS-Index.

        Args:
            chunks_file:   Pfad zur Chunks-JSON (auto-detect wenn None)
            force_rebuild: Index neu erstellen auch wenn bereits vorhanden
        """
        if chunks_file is None:
            chunks_file = self.project_root / 'data' / 'chunks' / f'{self.retrieval_mode}_chunks.json'

        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks-Datei nicht gefunden: {chunks_file}")

        self.load_embedding_model()

        index_file = self.index_dir / f'{self.retrieval_mode}_index.faiss'
        chunks_cache = self.index_dir / f'{self.retrieval_mode}_chunks.pkl'

        if not force_rebuild and index_file.exists() and chunks_cache.exists():
            print(f"\nLade vorhandenen Index...")
            self.index = faiss.read_index(str(index_file))
            with open(chunks_cache, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"  Index geladen: {len(self.chunks)} Chunks")
            return

        print(f"\nLade Chunks: {chunks_file.name}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"  {len(self.chunks)} Chunks geladen.")

        print(f"\nErstelle Embeddings...")
        texts = [c['text'] for c in self.chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        print(f"\nErstelle FAISS-Index...")
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"  FAISS-Index bereit: {self.index.ntotal} Vektoren")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))
        with open(chunks_cache, 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"  Index gespeichert: {index_file.relative_to(self.project_root)}")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieval mit mode-spezifischem Re-Ranking.

        Args:
            query: Suchanfrage
            k:     Anzahl zurückzugebender Chunks

        Returns:
            Liste von Chunk-Dictionaries mit Score-Feldern
        """
        if self.index is None:
            raise RuntimeError("Kein Index geladen. Zuerst load_chunks() aufrufen.")

        query_emb = self.embedding_model.encode([query])
        faiss.normalize_L2(query_emb)

        n_results = min(k * 2, self.index.ntotal)
        similarities, indices = self.index.search(query_emb, n_results)

        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            candidates.append({
                'id': str(chunk['id']),
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'base_similarity': float(similarities[0][i])
            })

        if self.retrieval_mode == 'hierarchical':
            return self._hierarchical_rerank(candidates, k)
        else:
            candidates.sort(key=lambda x: x['base_similarity'], reverse=True)
            for c in candidates[:k]:
                c['final_score'] = c['base_similarity']
                c['chapter_bonus'] = 0.0
            return candidates[:k]

    def _hierarchical_rerank(self, candidates: List[Dict], k: int) -> List[Dict]:
        """
        Re-Ranking für den hierarchischen Modus: Chunks aus dem häufigsten
        Kapitel unter den Top-5-Kandidaten erhalten einen Kohärenz-Bonus.
        """
        from collections import Counter

        chapters = [c['metadata'].get('chapter', '') for c in candidates[:5]]
        chapter_counts = Counter(chapters)
        most_common_chapter = chapter_counts.most_common(1)[0][0] if chapter_counts else None

        for candidate in candidates:
            score = candidate['base_similarity']
            if most_common_chapter and candidate['metadata'].get('chapter') == most_common_chapter:
                chapter_bonus = self.hierarchical_boost * (chapter_counts[most_common_chapter] / 5)
                score += chapter_bonus
                candidate['chapter_bonus'] = chapter_bonus
            else:
                candidate['chapter_bonus'] = 0.0
            candidate['final_score'] = score

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates[:k]

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """
        Generiert eine Antwort auf Basis der retrieved Chunks.

        Args:
            query:          Benutzerfrage
            context_chunks: Retrieved Chunks als Kontext

        Returns:
            Dict mit 'answer', 'model', 'tokens'
        """
        self.load_llm()

        context = "\n\n".join(
            f"[Quelle {i}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks, 1)
        )

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Du bist ein hilfreicher Assistent für Teilnehmer der XML-Validierungs-Schulung in GEFEG.FX.

Beantworte Fragen basierend auf den bereitgestellten Schulungsunterlagen.
Antworte klar, präzise und verständlich auf Deutsch.
Wenn die Antwort nicht in den Unterlagen steht, sage das ehrlich.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Basierend auf folgenden Schulungsunterlagen:

{context}

Beantworte die Frage: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            answer = response

        return {'answer': answer, 'model': self.llm_model_name, 'tokens': len(outputs[0])}

    def answer_question(self, query: str, k: int = 5, include_chunks: bool = True) -> Dict:
        """
        End-to-End: Retrieve + Generate.

        Args:
            query:          Benutzerfrage
            k:              Anzahl Chunks für das Retrieval
            include_chunks: Retrieved Chunks in der Antwort zurückgeben

        Returns:
            Dict mit 'answer', 'mode', 'k' und optional 'retrieved_chunks'
        """
        retrieved = self.retrieve(query, k)
        result = self.generate_answer(query, retrieved)
        result['query'] = query
        result['mode'] = self.retrieval_mode
        result['k'] = k

        if include_chunks:
            result['retrieved_chunks'] = retrieved

        return result


def main():
    """Demo: RAG-System testen."""
    print("=" * 70)
    print("XML VALIDATION RAG – DEMO")
    print("=" * 70)

    print("\nRetrieval-Modus:")
    print("  1) flat")
    print("  2) hierarchical")
    choice = input("\nModus (1/2) [default: 1]: ").strip() or '1'
    mode = 'flat' if choice == '1' else 'hierarchical'

    rag = XMLValidationRAG(retrieval_mode=mode)
    rag.load_chunks()

    test_queries = [
        "Was ist Schema-Validierung?",
        "Wie validiere ich ein XML-Dokument?",
        "Was ist der Unterschied zwischen Well-Formed und Valid?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print(f"{'='*70}")

        result = rag.answer_question(query, k=5)

        print("\nRetrieved Chunks:")
        for j, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"\n  {j}. Score: {chunk['final_score']:.3f}")
            print(f"     Dokument: {chunk['metadata']['doc_filename']}")
            if chunk['metadata'].get('chapter'):
                print(f"     Kapitel:  {chunk['metadata']['chapter']}")
            print(f"     Text:     {chunk['text'][:100]}...")

        print(f"\nAntwort:\n  {result['answer']}")
        print(f"\n  (Tokens: {result['tokens']}, Modell: {result['model']})")

    print(f"\n{'='*70}")
    print("Demo abgeschlossen.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()