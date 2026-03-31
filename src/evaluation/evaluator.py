"""
evaluator.py – Retrieval-Evaluation für RAG-Systeme.

Berechnet Recall@k, Precision@k und MRR. Unterstützt mode-spezifische
Ground Truth (flat_chunks vs. hierarchical_chunks im Testset-Format):

  "ground_truth": {
      "flat_chunks": [12, 45, 67],
      "hierarchical_chunks": [23, 56, 89]
  }
"""

from typing import List, Dict, Set
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict


class RAGEvaluator:
    """Evaluiert die Retrieval-Performance eines RAG-Systems."""

    def __init__(self, testset_file: Path):
        self.testset_file = testset_file
        self.questions = self._load_testset()
        print(f"Testset geladen: {len(self.questions)} Fragen")
        self._validate_testset()

    def _load_testset(self) -> List[Dict]:
        with open(self.testset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['questions']

    def _validate_testset(self):
        issues = []
        for q in self.questions:
            gt = q.get('ground_truth', {})
            if 'flat_chunks' not in gt and 'hierarchical_chunks' not in gt:
                if 'relevant_chunks' in gt:
                    issues.append(f"Q{q['id']}: Altes Format (relevant_chunks)")
                else:
                    issues.append(f"Q{q['id']}: Keine Ground Truth vorhanden")

        if issues:
            print(f"\nTestset-Format-Probleme ({len(issues)} Fragen):")
            for issue in issues[:5]:
                print(f"  {issue}")
            if len(issues) > 5:
                print(f"  ... und {len(issues)-5} weitere")

    def calculate_recall_at_k(self, retrieved_ids: Set[int], ground_truth_ids: Set[int]) -> float:
        """Recall@k = |Retrieved ∩ GroundTruth| / |GroundTruth|"""
        if not ground_truth_ids:
            return 0.0
        return len(retrieved_ids & ground_truth_ids) / len(ground_truth_ids)

    def calculate_precision_at_k(self, retrieved_ids: Set[int], ground_truth_ids: Set[int], k: int) -> float:
        """Precision@k = |Retrieved ∩ GroundTruth| / k"""
        if k == 0:
            return 0.0
        return len(retrieved_ids & ground_truth_ids) / k

    def calculate_mrr(self, retrieved_ids: List[int], ground_truth_ids: Set[int]) -> float:
        """MRR = 1/rank des ersten relevanten Chunks (0 falls nicht gefunden)."""
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in ground_truth_ids:
                return 1.0 / rank
        return 0.0

    def get_ground_truth_for_mode(self, question: Dict, mode: str) -> List[int]:
        """Gibt die mode-spezifischen Ground-Truth-Chunk-IDs zurück."""
        gt = question.get('ground_truth', {})
        if mode == 'flat':
            chunk_ids = gt.get('flat_chunks', [])
        elif mode == 'hierarchical':
            chunk_ids = gt.get('hierarchical_chunks', [])
        else:
            raise ValueError(f"Unbekannter Modus: {mode}")
        # Fallback auf altes Format
        if not chunk_ids:
            chunk_ids = gt.get('relevant_chunks', [])
        return chunk_ids

    def evaluate_retrieval(self, retrieved_chunks: List[Dict], ground_truth_chunk_ids: List[int], k: int) -> Dict:
        """Berechnet Recall, Precision und MRR für ein einzelnes Retrieval-Ergebnis."""
        retrieved_ids = [int(c['id']) for c in retrieved_chunks[:k]]
        retrieved_set = set(retrieved_ids)
        ground_truth_set = set(ground_truth_chunk_ids)

        return {
            'recall': self.calculate_recall_at_k(retrieved_set, ground_truth_set),
            'precision': self.calculate_precision_at_k(retrieved_set, ground_truth_set, k),
            'mrr': self.calculate_mrr(retrieved_ids, ground_truth_set),
            'retrieved_ids': retrieved_ids,
            'ground_truth_ids': list(ground_truth_set),
            'overlap': list(retrieved_set & ground_truth_set)
        }

    def evaluate_system(self, rag_system, k_values: List[int] = [3, 5, 10], save_retrieved: bool = False) -> pd.DataFrame:
        """
        Evaluiert ein vollständiges RAG-System über alle Testfragen.

        Args:
            rag_system:     RAG-System mit retrieve()-Methode und retrieval_mode-Attribut
            k_values:       Liste der k-Werte
            save_retrieved: Retrieved-Chunk-IDs in Ergebnisse aufnehmen (erhöht Speicherbedarf)

        Returns:
            DataFrame mit einer Zeile pro (Frage, k).
        """
        results = []
        mode = rag_system.retrieval_mode
        skipped = 0

        print(f"\nEvaluiere Modus '{mode}': {len(self.questions)} Fragen, k={k_values}")

        for q_idx, question in enumerate(self.questions, 1):
            q_id = question['id']
            ground_truth_ids = self.get_ground_truth_for_mode(question, mode)

            if not ground_truth_ids:
                skipped += 1
                continue

            if q_idx % 10 == 0:
                print(f"  Fortschritt: {q_idx}/{len(self.questions)}")

            retrieved_chunks = rag_system.retrieve(question['question'], k=max(k_values))

            for k in k_values:
                metrics = self.evaluate_retrieval(retrieved_chunks, ground_truth_ids, k)

                result = {
                    'question_id': q_id,
                    'query': question['question'],
                    'type': question['type'],
                    'difficulty': question['difficulty'],
                    'mode': mode,
                    'k': k,
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'mrr': metrics['mrr'],
                    'num_ground_truth': len(ground_truth_ids),
                    'num_overlap': len(metrics['overlap'])
                }

                if save_retrieved:
                    result['retrieved_ids'] = metrics['retrieved_ids']
                    result['ground_truth_ids'] = metrics['ground_truth_ids']
                    result['overlap_ids'] = metrics['overlap']

                results.append(result)

        if skipped:
            print(f"  Übersprungen: {skipped} Fragen (keine Ground Truth für '{mode}')")
        print(f"  Abgeschlossen: {len(results)} Evaluierungen")

        return pd.DataFrame(results)

    def calculate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(['mode', 'k']).agg({
            'recall': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'mrr': ['mean', 'std']
        }).round(4)

    def calculate_stratified_stats(self, df: pd.DataFrame, by: str = 'type') -> pd.DataFrame:
        return df.groupby(['mode', 'k', by]).agg({
            'recall': 'mean',
            'precision': 'mean',
            'mrr': 'mean'
        }).round(4)

    def compare_modes(self, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """Vergleicht Flat vs. Hierarchical für ein bestimmtes k."""
        df_k = df[df['k'] == k].copy()
        comparison = df_k.pivot_table(index='question_id', columns='mode', values=['recall', 'precision', 'mrr'])

        for metric in ['recall', 'precision', 'mrr']:
            if 'flat' in df_k['mode'].values and 'hierarchical' in df_k['mode'].values:
                comparison[f'{metric}_diff'] = comparison[(metric, 'hierarchical')] - comparison[(metric, 'flat')]

        return comparison