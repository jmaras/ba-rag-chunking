"""
03_run_evaluation.py – Retrieval-Evaluation für beide Chunking-Strategien.

Pipeline:
  1. Testset laden (testset_50_questions.json)
  2. RAG-Systeme initialisieren (Flat + Hierarchical)
  3. Retrieval für alle Fragen durchführen
  4. Metriken berechnen (RAGEvaluator)
  5. Ergebnisse als CSV speichern
"""

from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == 'scripts' else script_dir
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from rag.xml_validation_rag import XMLValidationRAG
from evaluation.evaluator import RAGEvaluator


def main():
    print("=" * 70)
    print("RETRIEVAL EVALUATION – FLAT VS. HIERARCHICAL")
    print("=" * 70)

    testset_file = project_root / 'data' / 'testset' / 'testset_50_questions.json'
    results_dir = project_root / 'results' / 'evaluation'
    results_dir.mkdir(parents=True, exist_ok=True)

    if not testset_file.exists():
        print(f"\nFehler: Testset nicht gefunden: {testset_file}")
        return

    evaluator = RAGEvaluator(testset_file)
    k_values = [3, 5, 10]

    print(f"\nKonfiguration:")
    print(f"  Fragen:   {len(evaluator.questions)}")
    print(f"  k-Werte:  {k_values}")
    print(f"  Modi:     flat, hierarchical")

    print(f"\n{'='*70}")
    print(f"Geschätzte Laufzeit: ~2–5 Minuten")
    response = input(f"\nEvaluation starten? (j/n): ").strip().lower()

    if response not in ['j', 'ja', 'y', 'yes']:
        print("Abgebrochen.")
        return

    all_results = []

    # --- Flat ---
    print(f"\n{'='*70}")
    print("FLAT MODE")
    print(f"{'='*70}")

    try:
        flat_system = XMLValidationRAG(retrieval_mode='flat')
        flat_system.load_chunks()

        flat_results = evaluator.evaluate_system(flat_system, k_values=k_values, save_retrieved=False)
        all_results.append(flat_results)

        flat_summary = flat_results[flat_results['k'] == 5].agg({'recall': 'mean', 'precision': 'mean', 'mrr': 'mean'})
        print(f"\n  Recall@5:    {flat_summary['recall']:.4f}")
        print(f"  Precision@5: {flat_summary['precision']:.4f}")
        print(f"  MRR:         {flat_summary['mrr']:.4f}")

    except Exception as e:
        print(f"\nFehler (Flat): {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Hierarchical ---
    print(f"\n{'='*70}")
    print("HIERARCHICAL MODE")
    print(f"{'='*70}")

    try:
        hier_system = XMLValidationRAG(retrieval_mode='hierarchical')
        hier_system.load_chunks()

        hier_results = evaluator.evaluate_system(hier_system, k_values=k_values, save_retrieved=False)
        all_results.append(hier_results)

        hier_summary = hier_results[hier_results['k'] == 5].agg({'recall': 'mean', 'precision': 'mean', 'mrr': 'mean'})
        print(f"\n  Recall@5:    {hier_summary['recall']:.4f}")
        print(f"  Precision@5: {hier_summary['precision']:.4f}")
        print(f"  MRR:         {hier_summary['mrr']:.4f}")

    except Exception as e:
        print(f"\nFehler (Hierarchical): {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Ergebnisse speichern ---
    print(f"\n{'='*70}")
    print("ERGEBNISSE SPEICHERN")
    print(f"{'='*70}")

    combined_df = pd.concat(all_results, ignore_index=True)

    combined_df.to_csv(results_dir / 'results_combined.csv', index=False)
    flat_results.to_csv(results_dir / 'results_flat.csv', index=False)
    hier_results.to_csv(results_dir / 'results_hierarchical.csv', index=False)

    print(f"\n  results_combined.csv      ({len(combined_df)} Zeilen)")
    print(f"  results_flat.csv")
    print(f"  results_hierarchical.csv")

    # --- Kurzvergleich ---
    improvement = ((hier_summary['recall'] - flat_summary['recall']) / flat_summary['recall']) * 100
    print(f"\nRecall@5-Verbesserung (Hierarchical vs. Flat): {improvement:+.2f}%")

    # Metadaten speichern
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'testset': str(testset_file),
        'num_questions': len(evaluator.questions),
        'k_values': k_values,
        'flat_recall_mean': float(flat_summary['recall']),
        'hier_recall_mean': float(hier_summary['recall']),
        'improvement_pct': float(improvement)
    }

    with open(results_dir / 'evaluation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("Evaluation abgeschlossen.")
    print(f"{'='*70}")
    print(f"\nNächster Schritt: python scripts/04_analyze_results.py")


if __name__ == "__main__":
    try:
        import faiss
        import sentence_transformers
    except ImportError as e:
        print(f"\nFehlende Abhängigkeit: {e}")
        print("  pip install faiss-cpu sentence-transformers")
        sys.exit(1)

    main()