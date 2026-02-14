SHELL := /bin/zsh

.PHONY: setup smoke melbourne-full clean

setup:
	python3 -m venv .venv
	source .venv/bin/activate && pip install -r requirements.txt

smoke:
	./scripts/reproduce_melbourne_smoke.sh

melbourne-full:
	./scripts/reproduce_melbourne_full.sh

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".DS_Store" -type f -delete
