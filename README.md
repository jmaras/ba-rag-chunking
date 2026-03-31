# Structure-Aware vs. Flat Chunking für RAG-Systeme

Quellcode und Forschungsdaten zur Bachelorarbeit:

**„Structure-Aware vs. Flat Chunking für RAG-Systeme: Eine Evaluation am Beispiel von Schulungsunterlagen"**

Julian Maras · Matrikelnummer 102204481 · IU Internationale Hochschule · Wirtschaftsinformatik · 2026

## Überblick

Diese Arbeit vergleicht Flat Chunking (Sliding Window, 512 Tokens) mit Structure-Aware Chunking (Section-basierte Segmentierung mit Chapter Coherence Boost) für RAG-Systeme. Als Evaluationskorpus dienen fünf deutschsprachige Schulungsdokumente der GEFEG mbH zu den Themen Datenformate im EDI-Umfeld, XML-Grundlagen, Schema-Validierung, XML-Validierung sowie Einführung in GEFEG.FX.

## Installation

```bash
git clone https://github.com/[username]/ba-rag-chunking.git
cd ba-rag-chunking
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

Für die LLM-Inferenz wird eine NVIDIA GPU mit CUDA-Unterstützung benötigt. Für die LLM-as-Judge-Bewertung (Schritt 5) muss die Umgebungsvariable `OPENAI_API_KEY` gesetzt sein.

## Projektstruktur

```
ba-rag-chunking/
├── src/                        # Kernmodule (Parser, Chunker, RAG-Pipeline, Evaluator)
├── scripts/                    # Pipeline-Skripte (00–05, sequenziell)
├── notebooks/                  # Jupyter Notebooks (Exploration, Analyse, Judge)
├── data/
│   ├── raw/                    # Schulungsdokumente (NICHT ENTHALTEN)
│   ├── parsed/                 # Geparste Strukturen (NICHT ENTHALTEN)
│   ├── chunks/                 # Chunk-Dateien (NICHT ENTHALTEN)
│   ├── faiss_index/            # FAISS-Indizes (NICHT ENTHALTEN)
│   └── testset/                # Annotiertes Testset (50 Fragen mit Ground Truth)
├── results/
│   ├── evaluation/             # Retrieval-Metriken und statistische Tests
│   ├── llm_judge/              # LLM-as-Judge-Bewertungen
│   └── visualizations/         # 8 Abbildungen für die Thesis
├── requirements.txt
└── LICENSE
```

## Nicht enthaltene Dateien

Die Schulungsdokumente der GEFEG mbH (`data/raw/`), die daraus abgeleiteten Strukturen (`data/parsed/`, `data/chunks/`) sowie die FAISS-Indizes (`data/faiss_index/`) sind nicht im Repository enthalten. Die Schulungsunterlagen sind vertrauliche Unternehmensmaterialien; die abgeleiteten Strukturen und Indizes können ohne den Originalkorpus nicht reproduziert werden. Die Ergebnisse sind vollständig über die bereitgestellten CSV-Dateien und Visualisierungen in `results/` nachvollziehbar.

## Workflow

```bash
python scripts/00_check_setup.py        # Systemprüfung
python scripts/01_parse_documents.py    # DOCX → JSON (benötigt data/raw/)
python scripts/02_create_chunks.py      # Chunks erzeugen
python scripts/03_run_evaluation.py     # Retrieval-Evaluation
python scripts/04_analyze_results.py    # Statistische Auswertung
python scripts/05_llm_as_judge.py       # Antwortqualität bewerten
```

Schritte 1–3 erfordern den nicht enthaltenen Originalkorpus. Schritte 4 und 5 können mit den enthaltenen Ergebnisdaten direkt nachvollzogen werden.

## Zentrale Ergebnisse

| Metrik | Flat | Structure-Aware | Differenz | p-Wert |
|--------|------|-----------------|-----------|--------|
| Recall@5 | 0,342 | 0,453 | +32,4 % | 0,030 |
| MRR | 0,519 | 0,375 | −27,7 % | 0,013 |
| Korrelation Retrieval↔Antwort | 0,265 | 0,828 | — | — |

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe [LICENSE](LICENSE) für Details.