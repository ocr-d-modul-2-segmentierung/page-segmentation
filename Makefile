
.PHONY: release
release: requirements.txt
	pypi-release


requirements.txt: requirements.in
	pip-compile

.PHONY: requirements-devel
requirements-devel:
	pip install -r requirements-devel.txt 

