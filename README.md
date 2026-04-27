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