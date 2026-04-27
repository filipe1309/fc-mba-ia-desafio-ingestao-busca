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
    try:
        PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
            pre_delete_collection=True,
        )
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            print(f"\nErro: Cota da API excedida (rate limit).")
            print("Possíveis soluções:")
            print("  1. Aguarde a cota resetar (geralmente diária)")
            print("  2. Troque o provider no .env (PROVIDER=openai ou PROVIDER=gemini)")
            print("  3. Verifique seu plano em: https://ai.google.dev/gemini-api/docs/rate-limits")
        elif "404" in error_msg or "not found" in error_msg.lower():
            print(f"\nErro: Modelo de embedding não encontrado.")
            print("Verifique a variável GOOGLE_EMBEDDING_MODEL ou OPENAI_EMBEDDING_MODEL no .env")
        elif "auth" in error_msg.lower() or "api_key" in error_msg.lower() or "401" in error_msg:
            print(f"\nErro: Falha na autenticação com a API.")
            print("Verifique sua chave de API no .env (GOOGLE_API_KEY ou OPENAI_API_KEY)")
        else:
            print(f"\nErro ao gerar embeddings: {e}")
        sys.exit(1)

    print(f"Ingestão concluída! {len(chunks)} chunks armazenados na coleção '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()