"""
05_llm_as_judge.py – Qualitative Evaluation mit LLM-as-Judge (GPT-4o-mini).

Ablauf:
  1. Antworten mit beiden Strategien (Flat + Hierarchical) generieren
  2. GPT-4o-mini bewertet Qualität auf einer 1–5-Skala
  3. Vergleich: Führt besseres Retrieval zu besseren Antworten?

Bewertungskriterien:
  - Correctness  (Faktische Korrektheit)
  - Completeness (Vollständigkeit)
  - Relevance    (Direkter Bezug zur Frage)
"""

from pathlib import Path
import sys
import json
import pandas as pd
from typing import Dict, List
import os
from datetime import datetime
import time

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == 'scripts' else script_dir
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from rag.xml_validation_rag import XMLValidationRAG

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    print("OpenAI-Bibliothek nicht gefunden. Installieren: pip install openai")
    HAS_OPENAI = False

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=project_root / '.env')
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


class LLMJudge:
    """LLM-as-Judge Evaluator mit GPT-4o-mini."""

    def __init__(self, api_key: str = None):
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI-Bibliothek nicht installiert.")

        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = "gpt-4o-mini"
        print(f"LLM Judge bereit (Modell: {self.model})")

    def create_judge_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        return f"""Du bist ein Experte für XML-Validierung und evaluierst die Qualität von Antworten auf technische Fragen.

FRAGE:
{question}

REFERENZ-ANTWORT (Ground Truth):
{ground_truth}

ZU BEWERTENDE ANTWORT:
{answer}

AUFGABE:
Bewerte die Antwort nach folgenden Kriterien auf einer Skala von 1–5:

1. CORRECTNESS (Korrektheit):
   5: Vollständig korrekt  |  3: Teilweise korrekt  |  1: Komplett falsch

2. COMPLETENESS (Vollständigkeit):
   5: Alle Aspekte vollständig  |  3: Nur Kernaspekte  |  1: Sehr unvollständig

3. RELEVANCE (Relevanz):
   5: Direkt fokussiert  |  3: Teilweise relevant  |  1: Komplett irrelevant

Antworte NUR mit einem JSON-Objekt:
{{
  "correctness": <1-5>,
  "completeness": <1-5>,
  "relevance": <1-5>,
  "overall": <1-5>,
  "reasoning": "Kurze Begründung (max. 2 Sätze)"
}}"""

    def judge_answer(self, question: str, answer: str, ground_truth: str, max_retries: int = 3) -> Dict:
        prompt = self.create_judge_prompt(question, answer, ground_truth)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Du bist ein objektiver Evaluator für technische Antworten. Antworte NUR mit JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)
                required_keys = ['correctness', 'completeness', 'relevance', 'overall']

                if all(k in result for k in required_keys):
                    return result
                else:
                    print(f"  Versuch {attempt+1}: Fehlende Schlüssel in der Antwort.")

            except Exception as e:
                print(f"  Versuch {attempt+1} fehlgeschlagen: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        return {'correctness': 0, 'completeness': 0, 'relevance': 0, 'overall': 0, 'reasoning': 'Evaluation failed'}


def select_evaluation_subset(testset_file: Path, n_questions: int = 20, strategy: str = 'stratified') -> List[Dict]:
    """Wählt ein stratifiziertes oder zufälliges Subset aus dem Testset."""
    with open(testset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data['questions']

    if strategy == 'stratified':
        import random
        from collections import defaultdict

        groups = defaultdict(list)
        for q in questions:
            groups[(q['type'], q['difficulty'])].append(q)

        selected = []
        n_per_group = max(1, n_questions // len(groups))

        for group_questions in groups.values():
            selected.extend(random.sample(group_questions, min(n_per_group, len(group_questions))))

        if len(selected) < n_questions:
            remaining = [q for q in questions if q not in selected]
            selected.extend(random.sample(remaining, n_questions - len(selected)))

        if len(selected) > n_questions:
            selected = random.sample(selected, n_questions)

        return selected

    else:
        import random
        return random.sample(questions, n_questions)


def main():
    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 70)

    if not HAS_DOTENV:
        print("\npython-dotenv nicht installiert (empfohlen: pip install python-dotenv)")

    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("\nFehler: Kein OPENAI_API_KEY gefunden.")
        print("  1. .env-Datei im Projektverzeichnis erstellen")
        print("  2. Zeile einfügen: OPENAI_API_KEY=sk-proj-...")
        print("  3. pip install python-dotenv")
        return

    masked_key = api_key[:8] + "..." + api_key[-4:]
    print(f"\nAPI-Key geladen: {masked_key}")

    testset_file = project_root / 'data' / 'testset' / 'testset_50_questions.json'
    results_dir = project_root / 'results' / 'llm_judge'
    results_dir.mkdir(parents=True, exist_ok=True)

    if not testset_file.exists():
        print(f"\nFehler: Testset nicht gefunden: {testset_file}")
        return

    N_QUESTIONS = 20
    K_RETRIEVAL = 5

    print(f"\nKonfiguration:")
    print(f"  Subset-Größe: {N_QUESTIONS} Fragen")
    print(f"  Retrieval k:  {K_RETRIEVAL}")
    print(f"  Judge-Modell: gpt-4o-mini")

    selected_questions = select_evaluation_subset(testset_file, n_questions=N_QUESTIONS, strategy='stratified')
    print(f"\n{len(selected_questions)} Fragen ausgewählt.")

    from collections import Counter
    type_dist = Counter(q['type'] for q in selected_questions)
    diff_dist = Counter(q['difficulty'] for q in selected_questions)

    print(f"\n  Verteilung nach Typ:")
    for qtype, count in type_dist.items():
        print(f"    {qtype}: {count}")
    print(f"\n  Verteilung nach Schwierigkeit:")
    for diff, count in diff_dist.items():
        print(f"    {diff}: {count}")

    estimated_time = len(selected_questions) * 2 * 0.5
    print(f"\nGeschätzte Laufzeit: ~{estimated_time:.0f} Minuten")

    response = input(f"\nEvaluation starten? (j/n): ").strip().lower()
    if response not in ['j', 'ja', 'y', 'yes']:
        print("Abgebrochen.")
        return

    # Systeme initialisieren
    print(f"\n{'='*70}")
    print("SYSTEME INITIALISIEREN")
    print(f"{'='*70}")

    flat_system = XMLValidationRAG(retrieval_mode='flat')
    flat_system.load_chunks()

    hier_system = XMLValidationRAG(retrieval_mode='hierarchical')
    hier_system.load_chunks()

    flat_system.load_llm()
    # LLM zwischen beiden Systemen teilen (spart VRAM)
    hier_system.llm_model = flat_system.llm_model
    hier_system.llm_tokenizer = flat_system.llm_tokenizer

    judge = LLMJudge(api_key=api_key)

    # Evaluation
    print(f"\n{'='*70}")
    print("EVALUATION LÄUFT")
    print(f"{'='*70}")

    results = []

    for q_idx, question in enumerate(selected_questions, 1):
        q_id = question['id']
        query = question['question']
        ground_truth = question['ground_truth']['answer_summary']

        print(f"\n[{q_idx}/{len(selected_questions)}] Q{q_id}: {query[:55]}...")

        flat_answer = flat_system.answer_question(query, k=K_RETRIEVAL, include_chunks=False)['answer']
        hier_answer = hier_system.answer_question(query, k=K_RETRIEVAL, include_chunks=False)['answer']

        flat_scores = judge.judge_answer(query, flat_answer, ground_truth)
        hier_scores = judge.judge_answer(query, hier_answer, ground_truth)

        diff = hier_scores['overall'] - flat_scores['overall']
        label = f"Hierarchical besser (+{diff:.1f})" if diff > 0 else f"Flat besser ({diff:.1f})" if diff < 0 else "Gleich"
        print(f"  -> {label}")

        results.append({
            'question_id': q_id,
            'query': query,
            'type': question['type'],
            'difficulty': question['difficulty'],
            'ground_truth': ground_truth,
            'flat_answer': flat_answer,
            'flat_correctness': flat_scores['correctness'],
            'flat_completeness': flat_scores['completeness'],
            'flat_relevance': flat_scores['relevance'],
            'flat_overall': flat_scores['overall'],
            'flat_reasoning': flat_scores.get('reasoning', ''),
            'hier_answer': hier_answer,
            'hier_correctness': hier_scores['correctness'],
            'hier_completeness': hier_scores['completeness'],
            'hier_relevance': hier_scores['relevance'],
            'hier_overall': hier_scores['overall'],
            'hier_reasoning': hier_scores.get('reasoning', '')
        })

    # Ergebnisse speichern
    print(f"\n{'='*70}")
    print("ERGEBNISSE SPEICHERN")
    print(f"{'='*70}")

    df = pd.DataFrame(results)
    df.to_csv(results_dir / 'llm_judge_detailed.csv', index=False)

    with open(results_dir / 'llm_judge_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Aggregierte Statistiken
    print(f"\n{'='*70}")
    print("AGGREGIERTE STATISTIKEN")
    print(f"{'='*70}")

    metrics = ['correctness', 'completeness', 'relevance', 'overall']
    print(f"\n{'Metrik':<15} | {'Flat':>6} | {'Hier':>6} | {'Diff':>7}")
    print("-" * 48)

    summary_data = []
    for metric in metrics:
        flat_mean = df[f'flat_{metric}'].mean()
        hier_mean = df[f'hier_{metric}'].mean()
        diff = hier_mean - flat_mean

        print(f"{metric.capitalize():<15} | {flat_mean:>6.2f} | {hier_mean:>6.2f} | {diff:>+6.2f}")

        summary_data.append({
            'metric': metric,
            'flat_mean': flat_mean,
            'hier_mean': hier_mean,
            'difference': diff,
            'hier_better_count': (df[f'hier_{metric}'] > df[f'flat_{metric}']).sum(),
            'flat_better_count': (df[f'flat_{metric}'] > df[f'hier_{metric}']).sum(),
            'equal_count': (df[f'flat_{metric}'] == df[f'hier_{metric}']).sum()
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / 'llm_judge_summary.csv', index=False)

    # Win/Loss/Draw
    hier_wins = (df['hier_overall'] > df['flat_overall']).sum()
    flat_wins = (df['flat_overall'] > df['hier_overall']).sum()
    draws = (df['flat_overall'] == df['hier_overall']).sum()

    print(f"\nWin/Loss/Draw (Overall):")
    print(f"  Hierarchical: {hier_wins}/{len(df)} ({hier_wins/len(df)*100:.1f}%)")
    print(f"  Flat:         {flat_wins}/{len(df)} ({flat_wins/len(df)*100:.1f}%)")
    print(f"  Unentschieden: {draws}/{len(df)} ({draws/len(df)*100:.1f}%)")

    # Nach Kategorie
    print(f"\nNach Fragetyp:")
    print(df.groupby('type')[['flat_overall', 'hier_overall']].mean().round(2))

    print(f"\nNach Schwierigkeit:")
    print(df.groupby('difficulty')[['flat_overall', 'hier_overall']].mean().round(2))

    # Gesamtfazit
    overall_diff = summary_df[summary_df['metric'] == 'overall']['difference'].values[0]

    print(f"\n{'='*70}")
    if overall_diff > 0.5:
        print(f"Hierarchical erzielt deutlich bessere Antwortqualität (+{overall_diff:.2f} Punkte).")
    elif overall_diff > 0.2:
        print(f"Hierarchical erzielt moderat bessere Antwortqualität (+{overall_diff:.2f} Punkte).")
    elif overall_diff > -0.2:
        print(f"Beide Strategien erzielen vergleichbare Antwortqualität (Differenz: {overall_diff:+.2f}).")
    else:
        print(f"Flat erzielt bessere Antwortqualität (Differenz: {overall_diff:.2f} Punkte).")

    print(f"\nErgebnisse gespeichert in: results/llm_judge/")
    print(f"  llm_judge_detailed.csv")
    print(f"  llm_judge_summary.csv")
    print(f"  llm_judge_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    if not HAS_OPENAI:
        print("OpenAI-Bibliothek fehlt: pip install openai")
        sys.exit(1)

    if not HAS_DOTENV:
        print("Hinweis: python-dotenv nicht installiert (pip install python-dotenv).")
        input("Enter drücken zum Fortfahren...")

    main()