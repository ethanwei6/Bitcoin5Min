PYTHON ?= python3
PYTHONPATH := src
PYTHONPYCACHEPREFIX ?= work/pycache
CONFIG ?= config/paper_btc_5m.json
OUTPUT_DIR ?= outputs/paper_trader

.PHONY: test compile smoke run report postmortem dry-run clean

compile:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) -m compileall src scripts tests

test: compile
	PYTHONPATH=$(PYTHONPATH):tests PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/run_test_suite.py

smoke:
	PYTHONPATH=$(PYTHONPATH) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/run_paper_bot.py --config $(CONFIG) --settle-only --duration-seconds 0 --output-dir outputs/smoke

run:
	PYTHONPATH=$(PYTHONPATH) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/run_paper_bot.py --config $(CONFIG) --output-dir $(OUTPUT_DIR)

dry-run:
	PYTHONPATH=$(PYTHONPATH) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/run_execution_dry_run.py --config $(CONFIG) --side AUTO --spend-usd 5

report:
	PYTHONPATH=$(PYTHONPATH) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/run_daily_maintenance.py --output-dir $(OUTPUT_DIR) --report-dir $(OUTPUT_DIR)/reports

postmortem:
	PYTHONPATH=$(PYTHONPATH) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/analyze_paper_run.py --output-dir $(OUTPUT_DIR) --report-dir $(OUTPUT_DIR)/reports

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
