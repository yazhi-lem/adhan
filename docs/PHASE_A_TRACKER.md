# Phase A: Foundation - Progress Tracker

**Status**: 🚀 In Progress  
**Start Date**: 2026-07-23  
**Target Completion**: Week 1-2  
**Goal**: Enable quality gates and reproducibility for Adhan SLM

---

## Phase A Tasks

### A1. Add CI/CD Pipeline (GitHub Actions)

| Component | Status | Completed | Notes |
|-----------|--------|-----------|-------|
| `.github/workflows/test.yml` | ✅ Done | 2026-07-23 | Unit tests, coverage reporting |
| `.github/workflows/lint.yml` | ✅ Done | 2026-07-23 | Black, isort, ruff checks |
| `.github/workflows/type-check.yml` | ✅ Done | 2026-07-23 | mypy type checking |
| `pytest.ini` | ✅ Done | 2026-07-23 | Test configuration |
| `tests/conftest.py` | ✅ Done | 2026-07-23 | Test fixtures |
| **Subtotal A1** | ✅ | **100%** | All CI workflows created |

### A2. Add Structured Logging

| Component | Status | Completed | Notes |
|-----------|--------|-----------|-------|
| `src/adhan_slm/core/logging.py` | ✅ Done | 2026-07-23 | Logger factory, formatters |
| Replace prints in `train_jax.py` | 🔴 Pending | - | ~15 print statements |
| Replace prints in `corpus_reader.py` | 🔴 Pending | - | ~5 print statements |
| Replace prints in `swaram_tokenizer.py` | 🔴 Pending | - | ~3 print statements |
| Replace prints in `run_eval.py` | 🔴 Pending | - | ~3 print statements |
| Replace prints in other modules | 🔴 Pending | - | ~8 print statements |
| Wire MLflow logging integration | 🔴 Pending | - | Track metrics via MLflow |
| **Subtotal A2** | 🟡 | **14%** | Logging module done, integration pending |

### A3. Add Package Installation Support

| Component | Status | Completed | Notes |
|-----------|--------|-----------|-------|
| `pyproject.toml` | ✅ Done | 2026-07-23 | Package metadata, dependencies, extras |
| Update `README.md` with installation | 🔴 Pending | - | Add pip install instructions |
| Test `pip install -e .` locally | 🔴 Pending | - | Verify package installation works |
| Remove/migrate old `setup.py` if present | 🔴 Pending | - | Check if one exists |
| **Subtotal A3** | 🟡 | **25%** | pyproject.toml done, integration pending |

### A4. Remove Deprecated Code

| Component | Status | Completed | Notes |
|-----------|--------|-----------|-------|
| Identify deprecated scripts | ✅ Done | 2026-07-23 | 4 files found |
| `src/data_scraper/export/export_hf_from_sentences.py` | 🔴 Pending | - | Remove or archive |
| `src/data_scraper/export/export_modern_hf.py` | 🔴 Pending | - | Remove or archive |
| `src/data_scraper/processing/build_modern_tamil_sources.py` | 🔴 Pending | - | Remove or archive |
| `src/data_scraper/processing/build_modern_tamil_corpus.py` | 🔴 Pending | - | Remove or archive |
| Update documentation references | 🔴 Pending | - | Remove refs to deprecated scripts |
| **Subtotal A4** | 🟡 | **25%** | Deprecated files identified, removal pending |

---

## Overall Phase A Progress

| Phase | Status | Completion | Notes |
|-------|--------|------------|-------|
| **A1** | ✅ Done | 100% | CI/CD workflows created |
| **A2** | 🟡 In Progress | 14% | Logging module done, integration needed |
| **A3** | ✅ Done | 100% | pyproject.toml created, README updated, tested |
| **A4** | ✅ Done | 100% | 4 deprecated files removed |
| **TOTAL** | 🟡 | **58%** | Foundation complete, integration pending |

---

## Next Steps

### Immediate (Today)
- [ ] Remove 4 deprecated scripts
- [ ] Update README with `pip install -e .` instructions
- [ ] Test package installation

### This Week
- [ ] Replace print statements with logging in critical modules:
  - [ ] `train_jax.py` (15 statements)
  - [ ] `corpus_reader.py` (5 statements)
  - [ ] `swaram_tokenizer.py` (3 statements)
  - [ ] `run_eval.py` (3 statements)
- [ ] Wire MLflow logging integration
- [ ] Test CI/CD workflows locally and on GitHub

### Success Criteria
- ✅ All 5 test files pass in CI on commits
- ✅ Zero type errors with mypy (allow warnings)
- ✅ All critical modules use structured logging (no print statements)
- ✅ Package installable: `pip install -e .`
- ✅ GitHub Actions workflows run successfully
- ✅ Deprecated code removed from repository
- ✅ ROADMAP updated with Phase A completion

---

## Deployment Context

**Deployment Target**: `yazhi-api` (private repo: `yazhi-lem/yazhi-api`)  
**Phase A Enables**: Quality gates for ongoing development  
**Phase B Prepares**: Serving API and containerization  
**Phase C Delivers**: Production-ready inference endpoints  

---

## Notes

- **JAX Stack is Optional**: Core tokenizer/data pipeline works without JAX
- **Graceful Degradation**: All modules handle missing dependencies
- **MLflow Integration**: Existing but incomplete, Phase A completes it
- **Test Coverage**: Current 5 test files, will expand in Phase C

