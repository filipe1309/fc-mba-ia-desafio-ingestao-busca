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