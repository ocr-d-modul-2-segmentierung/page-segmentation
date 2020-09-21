
.PHONY: deps
deps: .deps

.deps: requirements.txt
	pip install -r requirements.txt
	touch .deps

.PHONY: release
release: requirements.txt
	pypi-release


requirements.txt: requirements.in .deps-devel
	pip-compile

.deps-devel: requirements-devel.txt
	pip install -r requirements-devel.txt
	touch .deps-devel

