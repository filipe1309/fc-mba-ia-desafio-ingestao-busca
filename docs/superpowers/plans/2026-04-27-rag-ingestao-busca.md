# RAG Ingestion & Semantic Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a RAG system that ingests a PDF into PostgreSQL/pgVector and answers questions via CLI using LangChain, supporting both OpenAI and Gemini providers.

**Architecture:** Procedural Python scripts (`config.py`, `ingest.py`, `search.py`, `chat.py`) with a shared config helper for provider selection. PostgreSQL with pgVector runs in Docker.

**Tech Stack:** Python, LangChain, PostgreSQL, pgVector, OpenAI/Gemini APIs, Docker Compose

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.env.example` | Modify | Add `PROVIDER`, `GEMINI_LLM_MODEL`, `OPENAI_LLM_MODEL` vars |
| `src/config.py` | Create | Provider selection, env loading, embedding/LLM/vector store factories |
| `src/ingest.py` | Modify | PDF loading, chunking, embedding, vector storage |
| `src/search.py` | Modify | Similarity search, prompt formatting, LLM call |
| `src/chat.py` | Modify | Interactive CLI loop |
| `Makefile` | Create | Convenience targets for setup, install, db, ingest, chat, run, stop |
| `README.md` | Modify | Execution instructions |

---

### Task 1: Update `.env.example` with all required variables

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Update `.env.example`**

Replace the current content with:

```env
PROVIDER=gemini

GOOGLE_API_KEY=
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GEMINI_LLM_MODEL=gemini-2.5-flash-lite

OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-5-nano

DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=rag_documents
PDF_PATH=document.pdf
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "chore: update .env.example with all required variables"
```

---

### Task 2: Create `src/config.py` — Provider factories

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Create `src/config.py`**

```python
import os
import sys

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "").lower()
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "rag_documents")
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


def get_embeddings():
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    print(f"Erro: PROVIDER inválido: '{PROVIDER}'. Use 'openai' ou 'gemini'.")
    sys.exit(1)


def get_llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
        return ChatOpenAI(model=model, temperature=0)

    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    print(f"Erro: PROVIDER inválido: '{PROVIDER}'. Use 'openai' ou 'gemini'.")
    sys.exit(1)


def get_vector_store():
    from langchain_postgres import PGVector

    embeddings = get_embeddings()
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/config.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add config module with provider factories"
```

---

### Task 3: Implement `src/ingest.py` — PDF ingestion

**Files:**
- Modify: `src/ingest.py`

- [ ] **Step 1: Replace `src/ingest.py` contents**

```python
import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from config import get_embeddings, DATABASE_URL, COLLECTION_NAME, PDF_PATH


load_dotenv()


def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Erro: arquivo PDF não encontrado: {PDF_PATH}")
        sys.exit(1)

    print(f"Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Páginas carregadas: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    print(f"Chunks criados: {len(chunks)}")

    print("Gerando embeddings e armazenando no banco de dados...")
    embeddings = get_embeddings()
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    print(f"Ingestão concluída! {len(chunks)} chunks armazenados na coleção '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/ingest.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/ingest.py
git commit -m "feat: implement PDF ingestion with chunking and vector storage"
```

---

### Task 4: Implement `src/search.py` — Search and LLM call

**Files:**
- Modify: `src/search.py`

- [ ] **Step 1: Replace `src/search.py` contents**

```python
from dotenv import load_dotenv

from config import get_vector_store, get_llm

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt():
    try:
        vector_store = get_vector_store()
        llm = get_llm()
    except Exception as e:
        print(f"Erro ao inicializar busca: {e}")
        return None

    def ask(question: str) -> str:
        results = vector_store.similarity_search_with_score(question, k=10)
        contexto = "\n\n".join(doc.page_content for doc, _score in results)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
        response = llm.invoke(prompt)
        return response.content

    return ask
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/search.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/search.py
git commit -m "feat: implement semantic search with similarity search and LLM call"
```

---

### Task 5: Implement `src/chat.py` — CLI loop

**Files:**
- Modify: `src/chat.py`

- [ ] **Step 1: Replace `src/chat.py` contents**

```python
from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Digite 'sair' para encerrar")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except KeyboardInterrupt:
            print("\nAté logo!")
            break

        if not question:
            continue

        if question.lower() == "sair":
            print("Até logo!")
            break

        response = chain(question)
        print(f"RESPOSTA: {response}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/chat.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/chat.py
git commit -m "feat: implement CLI chat loop with exit support"
```

---

### Task 6: Create `Makefile`

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Create `Makefile`**

```makefile
.PHONY: setup install db-up ingest chat run stop

setup:
	python3 -m venv venv

install:
	cp .env.example .env
	venv/bin/pip install -r requirements.txt

db-up:
	docker compose up -d

ingest:
	venv/bin/python src/ingest.py

chat:
	venv/bin/python src/chat.py

run: db-up setup install ingest chat

stop:
	docker compose down
```

- [ ] **Step 2: Verify Makefile syntax**

Run: `make -n run`
Expected: prints the commands that would be executed (dry run), no syntax errors

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile with setup, install, db, ingest, chat, run, stop targets"
```

---

### Task 7: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` contents**

```markdown
# Desafio MBA Engenharia de Software com IA - Full Cycle

## Ingestão e Busca Semântica com LangChain e PostgreSQL/pgVector

Sistema RAG (Retrieval-Augmented Generation) que ingere um PDF e responde perguntas via CLI usando apenas o conteúdo do documento como contexto.

### Tecnologias

- Python + LangChain
- PostgreSQL + pgVector (via Docker)
- OpenAI ou Gemini (configurável)

### Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API: OpenAI ou Google Gemini

### Execução rápida

```bash
# 1. Inicie tudo de uma vez
make run

# 2. Edite .env com sua chave de API e provider
#    PROVIDER=gemini (ou openai)
#    GOOGLE_API_KEY=sua-chave (ou OPENAI_API_KEY)

# 3. Execute a ingestão e o chat
make ingest
make chat
```

### Execução passo a passo

```bash
# Subir o banco de dados
make db-up

# Criar ambiente virtual
make setup

# Instalar dependências e copiar .env
make install

# Editar .env com suas credenciais
# vim .env

# Executar ingestão do PDF
make ingest

# Iniciar chat
make chat

# Parar tudo
make stop
```

### Variáveis de Ambiente

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `PROVIDER` | Provider LLM (`openai` ou `gemini`) | `gemini` |
| `GOOGLE_API_KEY` | Chave API Google | `AIza...` |
| `OPENAI_API_KEY` | Chave API OpenAI | `sk-...` |
| `DATABASE_URL` | String de conexão PostgreSQL | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PDF_PATH` | Caminho para o PDF | `document.pdf` |

### Estrutura do Projeto

```
├── Makefile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf
├── src/
│   ├── config.py      # Configuração e factories de providers
│   ├── ingest.py      # Ingestão do PDF
│   ├── search.py      # Busca semântica + prompt
│   └── chat.py        # CLI interativo
└── README.md
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with full execution instructions"
```

---

### Task 8: Integration test — Full pipeline

- [ ] **Step 1: Start database**

Run: `docker compose up -d`
Expected: PostgreSQL container starts and becomes healthy

- [ ] **Step 2: Verify database is ready**

Run: `docker compose ps`
Expected: `postgres_rag` container shows `healthy` status

- [ ] **Step 3: Run ingestion**

Run: `cd src && python ingest.py`
Expected output (similar to):
```
Carregando PDF: document.pdf
Páginas carregadas: N
Chunks criados: M
Gerando embeddings e armazenando no banco de dados...
Ingestão concluída! M chunks armazenados na coleção 'rag_documents'.
```

- [ ] **Step 4: Test chat with a relevant question**

Run: `cd src && python chat.py`
Type a question related to the PDF content.
Expected: receives a relevant answer from the LLM based on the PDF content.

- [ ] **Step 5: Test chat with an out-of-context question**

In the same chat session, type: `Qual é a capital da França?`
Expected response: `Não tenho informações necessárias para responder sua pergunta.`

- [ ] **Step 6: Test exit**

Type: `sair`
Expected: chat exits gracefully with `Até logo!`
