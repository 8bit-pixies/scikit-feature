install-python-ci-dependencies:
	pip install -e ".[ci]"


test-python:
	pytest tests

format-python:
	# Sort
	isort skfeature/ tests/

	# Format
	black --target-version py37 skfeature tests

lint-python:
	isort skfeature/ tests/ --check-only
	flake8 skfeature/ tests/
	black --check skfeature tests