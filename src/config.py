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

        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
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
