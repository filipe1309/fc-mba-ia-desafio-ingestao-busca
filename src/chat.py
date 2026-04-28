from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("\n" + "-"*80)
    print("3. Bem-vindo ao chat de busca! Faça suas perguntas sobre os documentos ingeridos.")
    print("> Digite 'sair' para encerrar")
    print("-"*80)

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
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
        print("-"*80)


if __name__ == "__main__":
    main()