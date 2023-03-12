.PHONY: build-wheel
build-wheel:
	@echo '[Building wheel]'
	@echo -n '[$$] '
	python3 setup.py bdist_wheel --universal

.PHONY: upload-wheel
upload-wheel:
	@echo '[Uploading wheel to pypi (requires twine)]'
	@echo -n '[$$] '
	python -m twine upload dist/* --skip-existing
