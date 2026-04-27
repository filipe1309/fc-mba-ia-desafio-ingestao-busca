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