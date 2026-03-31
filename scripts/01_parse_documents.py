"""
01_parse_documents.py – Parst alle Word-Dokumente und extrahiert strukturierte Daten.

Output: data/parsed/all_documents.json
"""

from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from parsing.docx_parser import DocxParser


def main():
    print("=" * 60)
    print("DOKUMENT-PARSER")
    print("=" * 60)

    parser = DocxParser()

    doc_paths = [
        project_root / 'data' / 'raw' / 'Datenformate-im-EDI-Umfeld.docx',
        project_root / 'data' / 'raw' / 'DE 2 Schema.docx',
        project_root / 'data' / 'raw' / 'DE 2 XML-Validierung.docx',
        project_root / 'data' / 'raw' / 'DE XMLBasics.docx',
        project_root / 'data' / 'raw' / 'EinführungGEFEG.FX.docx'
    ]

    output_file = project_root / 'data' / 'parsed' / 'all_documents.json'

    print(f"\nPrüfe Dokumente...")
    missing_docs = []
    for path in doc_paths:
        if path.exists():
            print(f"  OK  {path.name}")
        else:
            print(f"  --  {path.name} (nicht gefunden)")
            missing_docs.append(path.name)

    if missing_docs:
        print(f"\nFehler: {len(missing_docs)} Dokument(e) fehlen – alle Dateien müssen in data/raw/ liegen.")
        return

    print(f"\nAlle {len(doc_paths)} Dokumente gefunden.")

    print(f"\n{'='*60}")
    print("Konfiguration:")
    print(f"  Input:     {len(doc_paths)} .docx-Dateien")
    print(f"  Output:    {output_file.relative_to(project_root)}")
    print(f"  Hierarchie: 2 Ebenen (Kapitel + Abschnitte)")

    response = input(f"\nParsing starten? (j/n): ").strip().lower()
    if response not in ['j', 'ja', 'y', 'yes']:
        print("Abgebrochen.")
        return

    print(f"\n{'='*60}")
    print("Parsing läuft...")
    print(f"{'='*60}\n")

    try:
        all_docs = parser.parse_all_documents(doc_paths, output_file)

        print(f"\n{'='*60}")
        print("STATISTIKEN")
        print(f"{'='*60}")
        print(f"{'Dokument':<32} | {'K':>2} | {'S':>3} | {'P':>4} | {'Ø P/S'}")
        print("-" * 60)

        for doc in all_docs:
            stats = parser.get_statistics(doc)
            print(f"{stats['filename'][:30]:30} | "
                  f"{stats['num_chapters']:2} | "
                  f"{stats['num_sections']:3} | "
                  f"{stats['num_paragraphs']:4} | "
                  f"{stats['avg_paragraphs_per_section']:5.1f}")

        total_chapters = sum(len(d['chapters']) for d in all_docs)
        total_sections = sum(sum(len(ch['sections']) for ch in d['chapters']) for d in all_docs)
        total_paragraphs = sum(len(d['paragraphs']) for d in all_docs)

        print("-" * 60)
        print(f"{'GESAMT':30} | {total_chapters:2} | {total_sections:3} | {total_paragraphs:4} |")

        print(f"\nOutput gespeichert: {output_file}")
        print(f"Dateigröße: {output_file.stat().st_size / 1024:.1f} KB")

        print(f"\nNächster Schritt: python scripts/02_create_chunks.py")

    except Exception as e:
        print(f"\nFehler beim Parsing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()