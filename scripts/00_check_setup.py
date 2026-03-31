"""
00_check_setup.py – Dependency-Check
"""

from pathlib import Path
import sys

print("=" * 60)
print("DEPENDENCY CHECK")
print("=" * 60)

print(f"\nPython: {sys.version.split()[0]}")

deps = {
    'torch': 'torch',
    'transformers': 'transformers',
    'sentence_transformers': 'sentence-transformers',
    'faiss': 'faiss-cpu',
    'numpy': 'numpy',
    'bitsandbytes': 'bitsandbytes (optional, für GPU)'
}

missing = []
for module, package in deps.items():
    try:
        __import__(module)
        print(f"  OK  {package}")
    except ImportError:
        print(f"  --  {package} (nicht installiert)")
        missing.append(package)

if missing:
    print(f"\nFehlende Pakete:")
    for pkg in missing:
        print(f"  pip install {pkg.split()[0]}")
    sys.exit(1)

print(f"\n{'='*60}")
print("Alle Abhängigkeiten vorhanden.")
print(f"{'='*60}")

# FAISS testen
print("\nTeste FAISS...")

try:
    import faiss
    import numpy as np

    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    vectors = np.random.random((10, dimension)).astype('float32')
    faiss.normalize_L2(vectors)
    index.add(vectors)

    query = np.random.random((1, dimension)).astype('float32')
    faiss.normalize_L2(query)
    distances, indices = index.search(query, 3)

    print(f"  OK  FAISS funktioniert (Dimension: {dimension}, Vektoren: {index.ntotal})")

except Exception as e:
    print(f"  Fehler: {e}")
    sys.exit(1)

# Sentence-Transformers testen
print("\nTeste sentence-transformers...")

try:
    from sentence_transformers import SentenceTransformer

    print("  Lade Modell (beim ersten Mal kann das einen Moment dauern)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding = model.encode(["test sentence"])

    print(f"  OK  all-MiniLM-L6-v2 geladen (Embedding-Shape: {embedding.shape})")

except Exception as e:
    print(f"  Fehler: {e}")
    sys.exit(1)

# Transformers testen
print("\nTeste transformers...")

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer("test")
    print("  OK  transformers funktioniert")

except Exception as e:
    print(f"  Fehler: {e}")

print(f"\n{'='*60}")
print("Setup-Check abgeschlossen.")
print(f"{'='*60}")