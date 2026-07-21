PYTHON ?= python

.PHONY: check figure certificates certificate-grid limiting limiting-mc finite-mc selected-mc finite85 six-numerical six-mc six-certificate checksums

check:
	$(PYTHON) scripts/reproduce.py check

figure:
	$(PYTHON) scripts/reproduce.py figure

certificates:
	$(PYTHON) scripts/reproduce.py theorem-certificates

certificate-grid:
	$(PYTHON) scripts/reproduce.py certificate-grid

limiting:
	$(PYTHON) scripts/reproduce.py limiting-curve

limiting-mc:
	$(PYTHON) scripts/reproduce.py limiting-mc

finite-mc:
	$(PYTHON) scripts/reproduce.py finite-mc

selected-mc:
	$(PYTHON) scripts/reproduce.py selected-mc

finite85:
	$(PYTHON) scripts/reproduce.py finite-85

six-numerical:
	$(PYTHON) scripts/reproduce.py six-numerical

six-mc:
	$(PYTHON) scripts/reproduce.py six-mc

six-certificate:
	$(PYTHON) scripts/reproduce.py six-certificate

checksums:
	$(PYTHON) scripts/checksums.py --write
