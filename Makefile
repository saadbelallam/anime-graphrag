.PHONY: install db collect graph embed build ask eval diagnose

install:
	pip install -e .

db:
	docker compose up -d

collect:
	python scripts/collect.py

graph:
	python scripts/build_graph.py

embed:
	python scripts/embed.py

build: collect graph embed

eval:
	python -m eval.run_eval --failures

diagnose:
	python scripts/diagnose.py

app:
	streamlit run scripts/app.py
