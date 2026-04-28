.PHONY: setup install db-up ingest chat run stop

setup:
	@python3 -m venv venv

install:
	@test -f .env || cp .env.example .env
	@venv/bin/pip install -q -r requirements.txt

db-up:
	@docker compose up -d

ingest:
	@venv/bin/python src/ingest.py

chat:
	@venv/bin/python src/chat.py

run: db-up setup install ingest chat

stop:
	@docker compose down
