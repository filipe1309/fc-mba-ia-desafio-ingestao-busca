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