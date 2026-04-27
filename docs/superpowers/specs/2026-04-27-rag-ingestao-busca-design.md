# RAG Ingestion & Semantic Search — Design Spec

## Problem

Build a Retrieval-Augmented Generation (RAG) system that ingests a PDF into PostgreSQL with pgVector and answers user questions via CLI using only the PDF content as context.

## Approach

Procedural Python scripts following the challenge's required structure, with a lightweight `config.py` helper to avoid duplication of provider selection and DB connection logic. Supports both OpenAI and Gemini as configurable providers.

## Architecture

<!-- Data flow from PDF ingestion through to CLI answer -->

```mermaid
flowchart LR
    PDF["document.pdf"] -- "PyPDFLoader" --> Pages["Pages"]
    Pages -- "RecursiveCharacterTextSplitter\n(1000 / 150)" --> Chunks["Chunks"]
    Chunks -- "Embeddings API" --> Vectors["Vectors"]
    Vectors -- "PGVector" --> DB[("PostgreSQL\npgVector")]

    User["User (CLI)"] -- "question" --> Embed["Embed Question"]
    Embed -- "similarity_search\n(k=10)" --> DB
    DB -- "top 10 results" --> Prompt["Build Prompt"]
    Prompt -- "LLM call" --> Answer["Answer"]
    Answer -- "print" --> User
```

## Project Structure

```
├── docker-compose.yml          # PostgreSQL + pgVector
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── document.pdf                # Source PDF
├── src/
│   ├── config.py               # Provider selection, env loading, factories
│   ├── ingest.py               # PDF ingestion script
│   ├── search.py               # Search logic + prompt template
│   └── chat.py                 # CLI chat loop
└── README.md
```

## Configuration (`config.py`)

Loads `.env` via `python-dotenv` and exposes:

- **`get_embeddings()`** — returns `OpenAIEmbeddings` or `GoogleGenerativeAIEmbeddings` based on `PROVIDER`
- **`get_llm()`** — returns `ChatOpenAI` or `ChatGoogleGenerativeAI` based on `PROVIDER`
- **`get_vector_store()`** — returns a `PGVector` instance connected to the configured database and collection

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PROVIDER` | LLM provider (`openai` or `gemini`) | `gemini` |
| `GOOGLE_API_KEY` | Google AI API key | `AIza...` |
| `GOOGLE_EMBEDDING_MODEL` | Gemini embedding model | `models/embedding-001` |
| `GEMINI_LLM_MODEL` | Gemini LLM model | `gemini-2.5-flash-lite` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `OPENAI_LLM_MODEL` | OpenAI LLM model | `gpt-5-nano` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Vector store collection name | `rag_documents` |
| `PDF_PATH` | Path to the source PDF | `document.pdf` |

## Ingestion (`ingest.py`)

### Flow

1. Load PDF with `PyPDFLoader(PDF_PATH)`
2. Split into chunks: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`
3. Get embeddings via `config.get_embeddings()`
4. Store vectors with `PGVector.from_documents(chunks, embeddings, connection=DATABASE_URL, collection_name=...)`

### Behavior

- Prints progress: number of pages loaded, chunks created, vectors stored
- Overwrites existing collection on each run (fresh ingestion)
- Exits with error message if PDF not found or DB unreachable

## Search (`search.py`)

### Prompt Template

Already defined in the codebase. Uses the exact template from the challenge spec with `{contexto}` and `{pergunta}` placeholders.

### `search_prompt()` Function

Returns a callable that, given a question:

1. Initializes `PGVector` vector store (existing collection, read-only)
2. Runs `similarity_search_with_score(query, k=10)`
3. Concatenates the top results' `page_content` as context
4. Formats `PROMPT_TEMPLATE` with context and question
5. Calls the LLM via `config.get_llm()`
6. Returns the response string

## Chat (`chat.py`)

### Flow

1. Calls `search_prompt()` once at startup to initialize the chain
2. Prints `"Digite 'sair' para encerrar"` once at startup
3. Enters interactive loop:
   - Prints `"PERGUNTA: "` and reads user input
   - If input is `sair` → exits gracefully
   - Otherwise, calls the chain with the question
   - Prints `"RESPOSTA: {answer}"`
4. Handles `KeyboardInterrupt` (Ctrl+C) gracefully with a goodbye message

## Error Handling

- Missing `PROVIDER` env var → clear error message with valid options
- Invalid provider value → clear error message
- Missing API key for selected provider → clear error message
- PDF file not found → error with path shown
- Database connection failure → error with connection hint
- Empty search results → the prompt template handles this (LLM responds with "Não tenho informações")

## Execution Order

```bash
# 1. Start database
docker compose up -d

# 2. Activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with API keys and PROVIDER choice

# 4. Ingest PDF
python src/ingest.py

# 5. Start chat
python src/chat.py
```
