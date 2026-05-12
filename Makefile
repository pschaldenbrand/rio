.PHONY: format
format:
	uvx pre-commit run -a

.PHONY: type
type:
	uvx ty check

.PHONY: check
check: format type

.PHONY: test
test:
	uv run --extra dev pytest

.PHONY: test-unit
test-unit:
	uv run --extra dev pytest -m unit

.PHONY: test-integration
test-integration:
	uv run --extra dev pytest -m integration

.PHONY: test-gpu
test-gpu:
	uv run --extra dev pytest -m gpu

.PHONY: test-hardware
test-hardware:
	uv run --extra dev pytest -m hardware

.PHONY: build
build:
	uv build
