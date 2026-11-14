# Convenience wrappers. The primary entry point for users is ./run.sh
.PHONY: run lite setup health stop test docs check logs clean help
help:
	@echo "make run     set up if needed and start the app (same as ./run.sh)"
	@echo "make lite    same without PyTorch/neural models"
	@echo "make setup   create .venv and install dependencies only"
	@echo "make health  environment and pipeline smoke test (+ app, if running)"
	@echo "make stop    stop a running app"
	@echo "make test    run the test suite"
	@echo "make docs    check Markdown links and wording"
	@echo "make check   test + docs"
	@echo "make logs    follow the app log"
	@echo "make clean   remove runtime files (.rlx); keeps .venv and your library"
run:
	./run.sh
lite:
	./run.sh --lite
setup:
	scripts/setup.sh
health:
	scripts/healthcheck.sh
stop:
	scripts/stop.sh
test:
	.venv/bin/python -m pytest -q
docs:
	python3 scripts/check_docs.py
check: test docs
logs:
	tail -f .rlx/logs/app.log
clean:
	scripts/stop.sh --quiet; rm -rf .rlx
