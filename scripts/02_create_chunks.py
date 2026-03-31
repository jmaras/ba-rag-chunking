"""
02_create_chunks.py – Erstellt Chunks aus geparsten Dokumenten (token-basiert).
"""

from pathlib import Path
import json
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent if script_dir.name == 'scripts' else script_dir
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from chunking.flat_chunker import FlatChunker
from chunking.hierarchical_chunker import HierarchicalChunker


def main():
    print("=" * 60)
    print("CHUNKING PIPELINE (TOKEN-BASED)")
    print("=" * 60)

    parsed_file = project_root / 'data' / 'parsed' / 'all_documents.json'
    flat_output = project_root / 'data' / 'chunks' / 'flat_chunks.json'
    hier_output = project_root / 'data' / 'chunks' / 'hierarchical_chunks.json'

    if not parsed_file.exists():
        print(f"\nFehler: Input-Datei nicht gefunden: {parsed_file}")
        print("Zuerst ausführen: python scripts/01_parse_documents.py")
        return

    flat_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLade geparste Dokumente...")
    try:
        with open(parsed_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        print(f"  {len(docs)} Dokumente geladen.")
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # --- Flat Chunking ---
    print(f"\n{'='*60}")
    print("FLAT CHUNKING")
    print(f"{'='*60}")

    try:
        flat_chunker = FlatChunker(chunk_size=512, chunk_overlap=50)
        print(f"  Konfiguration: chunk_size={flat_chunker.chunk_size} Tokens, "
              f"overlap={flat_chunker.chunk_overlap} Tokens")

        flat_chunks = flat_chunker.chunk_documents(docs)
        flat_chunker.save_chunks(flat_chunks, str(flat_output))

        flat_stats = flat_chunker.get_statistics(flat_chunks)
        print(f"\n  Statistiken:")
        print(f"    Chunks:         {flat_stats['num_chunks']:,}")
        print(f"    Ø Tokens:       {flat_stats['avg_tokens']:.1f}")
        print(f"    Min/Max Tokens: {flat_stats['min_tokens']} / {flat_stats['max_tokens']}")

    except Exception as e:
        print(f"Fehler beim Flat-Chunking: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Hierarchical Chunking ---
    print(f"\n{'='*60}")
    print("HIERARCHICAL CHUNKING")
    print(f"{'='*60}")

    try:
        hier_chunker = HierarchicalChunker(max_chunk_size=512)
        print(f"  Konfiguration: max_chunk_size={hier_chunker.max_chunk_size} Tokens")

        hier_chunks = hier_chunker.chunk_documents(docs)
        hier_chunker.save_chunks(hier_chunks, str(hier_output))

        hier_stats = hier_chunker.get_statistics(hier_chunks)
        print(f"\n  Statistiken:")
        print(f"    Chunks:         {hier_stats['num_chunks']:,}")
        print(f"    Ø Tokens:       {hier_stats['avg_tokens']:.1f}")
        print(f"    Min/Max Tokens: {hier_stats['min_tokens']} / {hier_stats['max_tokens']}")
        print(f"    Gesplittet:     {hier_stats['num_split_chunks']}")
        print(f"    Ebenen:         {hier_stats['hierarchy_levels']}")

    except Exception as e:
        print(f"Fehler beim Hierarchical-Chunking: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Vergleich ---
    print(f"\n{'='*60}")
    print("VERGLEICH")
    print(f"{'='*60}")

    diff = len(flat_chunks) - len(hier_chunks)
    diff_pct = (diff / len(hier_chunks)) * 100 if len(hier_chunks) > 0 else 0

    print(f"  Flat chunks:         {len(flat_chunks):,}")
    print(f"  Hierarchical chunks: {len(hier_chunks):,}")
    print(f"  Differenz:           {diff:+,} ({diff_pct:+.1f}%)")
    print(f"\n  Ø Tokens pro Chunk:")
    print(f"    Flat:          {flat_stats['avg_tokens']:.1f}")
    print(f"    Hierarchical:  {hier_stats['avg_tokens']:.1f}")

    print(f"\n{'='*60}")
    print("Chunking abgeschlossen.")
    print(f"{'='*60}")


if __name__ == "__main__":
    print(f"Project Root: {project_root}")
    main()