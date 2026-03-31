"""
04_analyze_results.py – Statistische Analyse der Evaluationsergebnisse.

Erfordert: results_combined.csv (erstellt durch 03_run_evaluation.py)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == 'scripts' else script_dir
sys.path.insert(0, str(project_root))


def load_results(results_dir: Path = None):
    if results_dir is None:
        results_dir = project_root / 'results' / 'evaluation'

    combined_file = results_dir / 'results_combined.csv'

    if not combined_file.exists():
        print(f"Fehler: Ergebnisse nicht gefunden: {combined_file}")
        print("  Zuerst ausführen: python scripts/03_run_evaluation.py")
        return None

    df = pd.read_csv(combined_file)
    print(f"Ergebnisse geladen: {len(df)} Zeilen")
    return df


def statistical_tests(df: pd.DataFrame, k: int = 5):
    print(f"\n{'='*70}")
    print(f"STATISTISCHE TESTS (k={k})")
    print(f"{'='*70}")

    df_k = df[df['k'] == k].copy()

    flat_qids = set(df_k[df_k['mode'] == 'flat']['question_id'].unique())
    hier_qids = set(df_k[df_k['mode'] == 'hierarchical']['question_id'].unique())
    common_qids = flat_qids & hier_qids

    print(f"\nFragen-Abdeckung:")
    print(f"  Flat:             {len(flat_qids)}")
    print(f"  Hierarchical:     {len(hier_qids)}")
    print(f"  Gemeinsam (paired): {len(common_qids)}")

    if len(flat_qids) != len(hier_qids):
        missing_in_hier = flat_qids - hier_qids
        missing_in_flat = hier_qids - flat_qids
        if missing_in_hier:
            print(f"  Fehlt in hierarchical: {sorted(missing_in_hier)}")
        if missing_in_flat:
            print(f"  Fehlt in flat: {sorted(missing_in_flat)}")

    df_k = df_k[df_k['question_id'].isin(common_qids)].copy()

    flat = df_k[df_k['mode'] == 'flat'].sort_values('question_id')
    hier = df_k[df_k['mode'] == 'hierarchical'].sort_values('question_id')

    metrics = ['recall', 'precision', 'mrr']
    results = []

    for metric in metrics:
        flat_values = flat[metric].values
        hier_values = hier[metric].values

        t_stat, p_value = stats.ttest_rel(hier_values, flat_values)

        diff = hier_values - flat_values
        cohens_d = diff.mean() / diff.std()

        mean_diff = hier_values.mean() - flat_values.mean()
        pct_diff = (mean_diff / flat_values.mean()) * 100

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "(n.s.)"

        if abs(cohens_d) < 0.2:
            effect = "vernachlässigbar"
        elif abs(cohens_d) < 0.5:
            effect = "klein"
        elif abs(cohens_d) < 0.8:
            effect = "mittel"
        else:
            effect = "groß"

        print(f"\n{metric.upper()}:")
        print(f"  Flat:           {flat_values.mean():.4f} (±{flat_values.std():.4f})")
        print(f"  Hierarchical:   {hier_values.mean():.4f} (±{hier_values.std():.4f})")
        print(f"  Differenz:      {mean_diff:+.4f} ({pct_diff:+.2f}%)")
        print(f"  t-Statistik:    {t_stat:.4f}")
        print(f"  p-Wert:         {p_value:.4f} {sig}")
        print(f"  Cohen's d:      {cohens_d:.4f} ({effect})")

        results.append({
            'metric': metric,
            'flat_mean': flat_values.mean(),
            'flat_std': flat_values.std(),
            'hier_mean': hier_values.mean(),
            'hier_std': hier_values.std(),
            'diff': mean_diff,
            'diff_pct': pct_diff,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        })

    return pd.DataFrame(results)


def analyze_by_category(df: pd.DataFrame, k: int = 5):
    print(f"\n{'='*70}")
    print(f"PERFORMANCE NACH KATEGORIE (k={k})")
    print(f"{'='*70}")

    df_k = df[df['k'] == k].copy()

    print(f"\n-- Nach Fragetyp --")
    by_type = df_k.groupby(['mode', 'type'])['recall'].agg(['mean', 'std', 'count'])
    print(by_type.round(4))

    pivot_type = df_k.pivot_table(index='type', columns='mode', values='recall', aggfunc='mean')

    if 'flat' in pivot_type.columns and 'hierarchical' in pivot_type.columns:
        pivot_type['diff'] = pivot_type['hierarchical'] - pivot_type['flat']
        pivot_type['diff_pct'] = (pivot_type['diff'] / pivot_type['flat']) * 100
        print(f"\nVerbesserungen nach Fragetyp:")
        print(pivot_type[['flat', 'hierarchical', 'diff', 'diff_pct']].round(4))

    print(f"\n-- Nach Schwierigkeit --")
    by_diff = df_k.groupby(['mode', 'difficulty'])['recall'].agg(['mean', 'std', 'count'])
    print(by_diff.round(4))

    pivot_diff = df_k.pivot_table(index='difficulty', columns='mode', values='recall', aggfunc='mean')

    if 'flat' in pivot_diff.columns and 'hierarchical' in pivot_diff.columns:
        pivot_diff['diff'] = pivot_diff['hierarchical'] - pivot_diff['flat']
        pivot_diff['diff_pct'] = (pivot_diff['diff'] / pivot_diff['flat']) * 100
        print(f"\nVerbesserungen nach Schwierigkeit:")
        print(pivot_diff[['flat', 'hierarchical', 'diff', 'diff_pct']].round(4))


def find_interesting_cases(df: pd.DataFrame, k: int = 5, top_n: int = 5):
    print(f"\n{'='*70}")
    print(f"INTERESSANTE FÄLLE (k={k})")
    print(f"{'='*70}")

    df_k = df[df['k'] == k].copy()

    pivot = df_k.pivot_table(index='question_id', columns='mode', values='recall')

    if 'flat' not in pivot.columns or 'hierarchical' not in pivot.columns:
        print("Kein Vergleich möglich – fehlende Modi.")
        return

    pivot['diff'] = pivot['hierarchical'] - pivot['flat']

    question_info = df_k[['question_id', 'query', 'type', 'difficulty']].drop_duplicates()
    pivot = pivot.merge(question_info, on='question_id')

    print(f"\n-- Top {top_n}: Hierarchical >> Flat --")
    for _, row in pivot.nlargest(top_n, 'diff').iterrows():
        print(f"\n  Q{row['question_id']}: {row['query'][:65]}...")
        print(f"    Typ: {row['type']}, Schwierigkeit: {row['difficulty']}")
        print(f"    Flat: {row['flat']:.3f}  |  Hier: {row['hierarchical']:.3f}  |  Diff: {row['diff']:+.3f}")

    print(f"\n-- Top {top_n}: Flat >> Hierarchical --")
    for _, row in pivot.nsmallest(top_n, 'diff').iterrows():
        print(f"\n  Q{row['question_id']}: {row['query'][:65]}...")
        print(f"    Typ: {row['type']}, Schwierigkeit: {row['difficulty']}")
        print(f"    Flat: {row['flat']:.3f}  |  Hier: {row['hierarchical']:.3f}  |  Diff: {row['diff']:+.3f}")

    print(f"\n-- Top {top_n}: Beide schwach (max Recall < 0.3) --")
    both_bad = pivot[pivot[['flat', 'hierarchical']].max(axis=1) < 0.3].nlargest(top_n, 'diff')

    if len(both_bad) > 0:
        for _, row in both_bad.iterrows():
            print(f"\n  Q{row['question_id']}: {row['query'][:65]}...")
            print(f"    Typ: {row['type']}, Schwierigkeit: {row['difficulty']}")
            print(f"    Flat: {row['flat']:.3f}  |  Hier: {row['hierarchical']:.3f}")
    else:
        print("  Keine gefunden (alle Fragen erzielen akzeptablen Recall).")


def generate_thesis_summary(df: pd.DataFrame, output_file: Path = None):
    print(f"\n{'='*70}")
    print("THESIS SUMMARY TABLE")
    print(f"{'='*70}")

    summary_data = []

    for k in [3, 5, 10]:
        df_k = df[df['k'] == k]
        for mode in ['flat', 'hierarchical']:
            mode_data = df_k[df_k['mode'] == mode]
            summary_data.append({
                'k': k,
                'mode': mode,
                'recall_mean': mode_data['recall'].mean(),
                'recall_std': mode_data['recall'].std(),
                'precision_mean': mode_data['precision'].mean(),
                'precision_std': mode_data['precision'].std(),
                'mrr_mean': mode_data['mrr'].mean(),
                'mrr_std': mode_data['mrr'].std()
            })

    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.round(4))

    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"\nGespeichert: {output_file}")

    return summary_df


def main():
    print("=" * 70)
    print("ERGEBNISANALYSE")
    print("=" * 70)

    df = load_results()
    if df is None:
        return

    print(f"\nDatensatz:")
    print(f"  Zeilen:    {len(df)}")
    print(f"  Fragen:    {df['question_id'].nunique()}")
    print(f"  Modi:      {list(df['mode'].unique())}")
    print(f"  k-Werte:   {sorted(df['k'].unique())}")

    output_dir = project_root / 'results' / 'evaluation'

    stats_df = statistical_tests(df, k=5)
    stats_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print(f"\nGespeichert: statistical_tests.csv")

    analyze_by_category(df, k=5)
    find_interesting_cases(df, k=5, top_n=5)
    generate_thesis_summary(df, output_file=output_dir / 'thesis_summary_table.csv')

    print(f"\n{'='*70}")
    print("Analyse abgeschlossen.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()