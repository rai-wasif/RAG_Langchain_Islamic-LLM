# IlmGPT - Quran & Hadith RAG Assistant

IlmGPT is a Retrieval-Augmented Generation (RAG) project that answers Islamic questions using only retrieved Quran and Hadith context, with bilingual output (English + Urdu) and visible source citations.

## Project Purpose

The goal is to reduce hallucinations for religion-domain Q&A by grounding every answer in local, indexed text sources.

Core design principles:

- Retrieval-first (no free-form religious claims)
- Source-cited output
- Bilingual response format (English + Urdu)
- Local vector database for reproducible retrieval

## What Is Implemented

## LangChain Concepts Implemented

This project uses LangChain as the orchestration layer for the full RAG lifecycle, not only prompt calling.

### Core LangChain primitives used

- `Document` objects for normalized Quran/Hadith records with citation metadata
- `RecursiveCharacterTextSplitter` for chunking mixed short/long Islamic texts
- Message-based invocation (`HumanMessage`) for structured LLM interaction

### RAG flow implemented with LangChain

1. `CSV -> Document`  
Each dataset row is converted into a LangChain `Document` with strict citation metadata.

2. `Document -> Chunks`  
`RecursiveCharacterTextSplitter` creates retrieval-friendly segments while preserving context.

3. `Chunks -> Embeddings`  
Chunk text is vectorized and persisted to Chroma.

4. `Query -> Retrieval`  
The user query is embedded and matched in vector space, with optional source filtering.

5. `Retrieved Context -> Prompt`  
Retrieved chunks are formatted into numbered source blocks for grounded answer generation.

6. `Prompt -> LLM -> Rendered Answer`  
Model output is rendered in English and Urdu with source markers.

### Why LangChain matters in this repository

- Modular pipeline design from ingestion to answer rendering
- Clear extension path for retrievers, rerankers, and new corpora
- Better traceability and maintainability for a citation-first RAG system

### 1) Data and ingestion strategy

Primary files used by the pipeline:

- `Data/Quran Multiple Language/en.yusufali.csv`
- `Data/Hadees/all_hadiths_clean.csv`

Current dataset sizes in this repo:

- Quran rows: `6236`
- Hadith rows: `34441`
- Combined source rows: `40677`

Ingestion approach:

- Each Quran ayah becomes one LangChain `Document` with metadata (`surah`, `ayah`, `reference`, `type=quran`)
- Each Hadith record becomes one LangChain `Document` with metadata (`collection`, `book`, `number`, `reference`, `type=hadith`)
- Metadata is intentionally kept rich to support transparent citations in final answers

### 2) Chunking strategy

Chunking is done with `RecursiveCharacterTextSplitter`:

- `chunk_size=500`
- `chunk_overlap=100`
- Separators: `"\n\n", "\n", ". ", " ", ""`

Why this setup:

- Ayat are generally short, but many hadith entries are long
- Overlap preserves continuity so meaning is not lost across chunk boundaries

### 3) Embedding model and vector space

Embedding model:

- `sentence-transformers/all-MiniLM-L6-v2`

Embedding details:

- 384-dimensional vectors
- Fast CPU inference and strong semantic retrieval for short religious text segments

Vector DB:

- Chroma persistent store at `db/chroma`
- Collection name: `ilmgpt_collection`
- Similarity metric: cosine (`hnsw:space=cosine`)

Current persisted vector count:

- `55818` chunks indexed in Chroma

### 4) Retrieval and answer strategy

Retriever behavior:

- Query is embedded and matched against Chroma vectors
- Supports filters: Quran only, Hadith only, or both
- `top_k` is user-adjustable in the app (3 to 10)

App retrieval enhancement:

- Includes query expansion terms for topics like `salah`, `forgiveness`, `patience`, `charity`, and related Urdu/Arabic keywords to improve recall

Prompting strategy:

- Strict anti-hallucination instructions
- Forbids adding external information
- Requires citation markers (`SOURCE_n`) for each claim
- Includes fallback message when sources do not directly answer

### 5) LLM providers used

Notebook pipeline model:

- Gemini `gemini-2.0-flash`

Streamlit app providers:

- Groq with `llama-3.3-70b-versatile`
- Gemini with `gemini-2.5-flash`

## File-by-File Breakdown

- `ilmgpt_notebook.ipynb`
Purpose: end-to-end RAG build notebook.
Implements: data loading, document creation, chunking, embedding generation, Chroma persistence, retrieval tests, bilingual prompting, Gemini invocation.

- `streamlit_app.py`
Purpose: production-style interactive UI.
Implements: provider selection (Groq/Gemini), source filter selection, top-k slider, retrieval, strict prompt generation, bilingual rendering, source chips, and themed UX.

- `README.md`
Purpose: architecture and operational documentation.

- `requirements.txt`
Purpose: pinned dependencies for reproducible setup.

- `pyproject.toml`
Purpose: project metadata.

- `main.py`
Purpose: minimal placeholder entrypoint (currently a simple hello-world).

- `LINKEDIN_POST.txt`
Purpose: project summary draft for social sharing.

- `db/chroma/`
Purpose: persisted vector database artifacts (`chroma.sqlite3` + HNSW index files).

- `Data/`
Purpose: local Quran/Hadith CSV sources used during indexing.

## Repository Structure

```text
Quran-Hadees-RAG_Langchain_Project/
	ilmgpt_notebook.ipynb
	streamlit_app.py
	main.py
	requirements.txt
	pyproject.toml
	README.md
	LINKEDIN_POST.txt
	Data/
		Hadees/all_hadiths_clean.csv
		Quran Multiple Language/en.yusufali.csv
		Quran Multiple Language/Arabic-Original.csv
		Quran Multiple Language/Dutch.csv
		Quran Multiple Language/English.csv
		Quran Multiple Language/Urdu.csv
	db/chroma/
		chroma.sqlite3
		<hnsw index files>
```

## Setup

### 1. Create and activate environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API keys

Use one or both:

- `GROQ_API_KEY`
- `GOOGLE_API_KEY`

PowerShell example:

```powershell
$env:GROQ_API_KEY="your_key"
$env:GOOGLE_API_KEY="your_key"
```

## Build or Rebuild the Index

Run all relevant cells in `ilmgpt_notebook.ipynb` to recreate embeddings and Chroma collection.

## Run the App

```powershell
streamlit run streamlit_app.py
```

Open: `http://localhost:8501`

## Example Questions

- What does Islam say about patience?
- What is the importance of Salah?
- What does Quran say about forgiveness?
- What did the Prophet say about honesty?
- نماز کی فضیلت کیا ہے؟

## Output Screenshots (Placeholders)

<p align="center">
	<img src="" alt="Screenshot 1 - Home / Ask Screen" width="31%" />
	<img src="" alt="Screenshot 2 - English + Urdu Answer" width="31%" />
	<img src="" alt="Screenshot 3 - Retrieved Sources Panel" width="31%" />
</p>

<p align="center">
	<sub>Reserved space for 3 output screenshots. Add image paths when ready.</sub>
</p>

## Security and Operational Notes

- Never commit real API keys in code, notebook cells, or screenshots.
- This is an educational assistant and not a replacement for qualified scholarly verdicts.
- For public GitHub hosting, large data and DB artifacts may exceed practical repo size; use release assets, Git LFS, or external storage if needed.

## Disclaimer

IlmGPT is an educational tool. For critical religious matters, consult a qualified Islamic scholar.

یہ ایک تعلیمی ٹول ہے۔ اہم دینی معاملات میں مستند عالم سے رجوع کریں۔

## Author

Muhammad Wasif