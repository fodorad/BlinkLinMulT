# Contributing to BlinkLinMulT

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Quick start

```bash
git clone https://github.com/fodorad/BlinkLinMulT
cd BlinkLinMulT
uv sync --extra dev --extra docs
uv run pre-commit install   # optional: runs ruff automatically before every commit
```

To run the training, preprocessing, and experiment scripts, install the `train` extra:

```bash
uv sync --extra train        # or: uv sync --all-extras
```

Use `uv` (not plain `pip`) for the `train` extra — it applies the
`[tool.uv] override-dependencies` needed to resolve the
exordium/linmult/blinklinmult stack. This creates/updates the project venv at
`.venv` (its activation prompt shows `(blinklinmult)`); run tools with
`uv run <cmd>` or `source .venv/bin/activate`.

---

## Development workflow

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes** — keep them focused and minimal.
3. **Write or update tests** in `tests/` to cover your changes.
4. **Run checks locally** before pushing:

   ```bash
   make fix    # auto-format and fix lint issues
   make check  # lint + type-check + tests + docs build (mirrors CI)
   ```

5. **Open a Pull Request** against `main` and fill in the template.

---

## Commit message convention

BlinkLinMulT follows **Conventional Commits** so that
[release-please](https://github.com/googleapis/release-please) can generate the
changelog and the correct version bump automatically.

| Prefix | Meaning | Version bump |
|--------|---------|--------------|
| `fix:` | Bug fix, regression, hotfix | **Patch** (3.0.x) |
| `feat:` | New feature, new config option | **Minor** (3.x.0) |
| `feat!:` or `BREAKING CHANGE:` | API or config change that breaks existing usage | **Major** (x.0.0) |
| `docs:` | Documentation only | No bump |
| `test:` | Tests only | No bump |
| `refactor:` | Code refactor with no behaviour change | No bump |
| `chore:` | Build, CI, dependency updates | No bump |

### Examples

```
fix: correct the eye-state label polarity for MRL-Eye
feat: add a sliding-window evaluation pass for HUST-LEBW
feat!: replace the 1.x models API with config-driven training
docs: add the dataset rebuild recipe
chore: bump ruff to v0.9
```

---

## Release process

Releases are **automated** via [release-please](https://github.com/googleapis/release-please).

1. Merge Conventional-Commit PRs into `main`.
2. release-please opens / updates a **release PR** that bumps the version and
   updates `CHANGELOG.md`.
3. Merging that release PR creates the git **tag** (e.g. `v2.2.0`) and a GitHub
   Release.
4. The tag triggers the **CD** workflow, which builds the wheel (version derived
   from the tag via `hatch-vcs`) and publishes it to PyPI.

There is no version number stored in `pyproject.toml` and no manual "bump
version" commit to make.

---

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/) — run `make fix` to auto-apply.
- **Type checker**: [ty](https://github.com/astral-sh/ty) — run `make type-check`.
- **Line length**: 100 characters.
- **Python version**: 3.13+.
- **Docstrings**: Google style.

---

## Tests

```bash
make test              # run all tests with coverage
```

Test files live in `tests/` and mirror the `blinklinmult/` package
structure. Use small random tensors to assert expected input/output shapes.

---

## Reporting bugs and requesting features

Please use the GitHub issue templates:

- **Bug report**: include a minimal reproducible example, Python/PyTorch version, and OS.
- **Feature request**: describe the problem you are trying to solve, not just the solution.

---

## License

By contributing you agree that your work will be released under the
[MIT License](LICENSE).
