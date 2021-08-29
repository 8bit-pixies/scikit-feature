
test-python:
	pytest -n 8 tests

format-python:
	# Sort
	isort skfeature/ tests/

	# Format
	black --target-version py37 skfeature tests

lint-python:
	isort skfeature/ tests/ --check-only
	flake8 skfeature/ tests/
	black --check skfeature tests