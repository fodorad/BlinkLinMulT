"""Sphinx configuration for the BlinkLinMulT documentation."""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

project = "BlinkLinMulT"
copyright = "2023-2026, Ádám Fodor"
author = "Ádám Fodor"
release = "latest"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# ── sphinx-autoapi ─────────────────────────────────────────────────────────────

autoapi_dirs = ["../blinklinmult"]
autoapi_ignore = [
    # The preprocess scripts are standalone research CLIs that import the heavy
    # optional extraction stack; they are excluded from the wheel and are not
    # part of the published library API.
    "*/preprocess/*",
]
autoapi_options = [
    "members",
    "undoc-members",  # required -- without this autoapi skips submodule pages
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_add_toctree_entry = False
autoapi_member_order = "source"
autoapi_python_class_content = "both"


def skip_undocumented_attributes(app, what, name, obj, skip, options):
    """Hide undocumented attributes (instance variables) from the API docs.

    Keeps every class, function, method, and module even without a docstring,
    but hides bare attributes that have none (e.g. ``self.dropout = ...``).
    """
    if what == "attribute" and not obj.docstring:
        return True
    return skip


# ── The README is the front page ───────────────────────────────────────────────
# ``index.md`` includes ``../README.md`` verbatim rather than restating it, so
# the repository front page and the documentation front page cannot drift apart.
#
# The cost of that choice is link translation. The README's relative links are
# written from the repository root, where GitHub and PyPI resolve them; from
# inside the built site the same targets live elsewhere. The maps below drive a
# build-time rewrite of a generated copy, which keeps the README itself free of
# any Sphinx-specific markup.


README_PAGE_LINKS = {
    "docs/comparison.md": "comparison",
    "docs/data.md": "data",
    "docs/training.md": "training",
    "docs/inference.md": "inference",
    "docs/frame_wise.md": "frame_wise",
    "docs/eye_state_events.md": "eye_state_events",
    "docs/migration.md": "migration",
}
"""Repo-relative README links, mapped to the docname they become on the site."""

REPO_BLOB = "https://github.com/fodorad/BlinkLinMulT/blob/main/"
"""Base for README links to files that are not pages of this site."""


GENERATED_README = "_readme.md"
"""Filename of the build-time copy of the README, included by ``index.md``.

Generated on every build and never edited by hand. ``index.md`` includes this
rather than ``../README.md`` because the ``include`` directive reads its file
straight from disk, so a ``source-read`` handler never sees the README's text
and its links cannot be translated in place.
"""


def _translate_link(match: "re.Match[str]") -> str:
    """Translate one markdown link target from repo-relative to site-relative.

    Args:
        match (re.Match[str]): A ``[label](target)`` match.

    Returns:
        str: The link, with its target repointed if it needed it.
    """
    label, target = match.group(1), match.group(2)
    if "://" in target or target.startswith(("#", "mailto:")):
        return match.group(0)

    path, _, anchor = target.partition("#")
    page = README_PAGE_LINKS.get(path)
    if page is None:
        return f"[{label}]({REPO_BLOB}{target})"
    return f"[{label}]({page}.md" + (f"#{anchor}" if anchor else "") + ")"


def _drop_sourceless_viewcode_modules(app, env) -> None:
    """Remove modules viewcode could not find source for, before it builds its index.

    ``sphinx.ext.viewcode`` records ``False`` for a module whose source it cannot
    read -- ``builtins`` and friends are written in C, so there is no ``.py`` to
    highlight. That correctly suppresses the module's page, but the index page is
    built from the mapping's keys without skipping those entries, so it links a
    page that was deliberately never written.

    Dropping the falsy entries leaves the index listing only modules that really
    have a page. Without it the build emits a dead ``builtins.html`` link.

    Args:
        app (Sphinx): The running application.
        env (BuildEnvironment): The environment holding viewcode's mapping.
    """
    modules = getattr(env, "_viewcode_modules", None)
    if not modules:
        return
    for name in [name for name, entry in modules.items() if not entry]:
        del modules[name]


def _generate_readme_page(app):
    """Write the link-corrected copy of the README that ``index.md`` includes.

    Args:
        app (Sphinx): The running application, for its source directory.
    """
    readme = Path(app.srcdir).parent / "README.md"
    # `[^\]]*` would stop at the inner `]` of a badge (`[![alt](img)](target)`),
    # leaving the outer target untranslated; `(?:[^\[\]]|\[[^\]]*\])*` allows one
    # level of nesting so badge links are rewritten too.
    pattern = r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]+)\)"
    text = re.sub(pattern, _translate_link, readme.read_text())
    (Path(app.srcdir) / GENERATED_README).write_text(_with_one_title(text))


def _with_one_title(text: str) -> str:
    """Give the README a single H1 title, demoting its section headings.

    The README uses ``#`` for each top-level section, which reads well on GitHub
    but leaves the page with fourteen H1s and no title. Sphinx builds both the
    page title and the sidebar tree from the heading levels, so every section is
    pushed down one level and one title is added above them.

    Headings inside fenced code blocks are left alone.

    Args:
        text (str): The README's markdown.

    Returns:
        str: The markdown with exactly one H1 at the top.
    """
    lines, fenced = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("#"):
            line = "#" + line
        lines.append(line)
    return f"# {project}\n\n" + "".join(lines)


def setup(app):
    """Connect the custom build handlers."""
    app.connect("autoapi-skip-member", skip_undocumented_attributes)
    _generate_readme_page(app)
    app.connect("env-updated", _drop_sourceless_viewcode_modules)


# ── Napoleon (Google docstrings) ───────────────────────────────────────────────

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── MyST ───────────────────────────────────────────────────────────────────────

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# ── Furo theme ─────────────────────────────────────────────────────────────────
# Furo follows the OS prefers-color-scheme; the dark/light toggle is built in.

# Sources for the Hugging Face cards, not pages of this site. Each has a
# repo-facing counterpart here: data.md for the dataset, the README for the
# models. They live under docs/ so the published cards and this site are built
# from one text rather than drifting apart.
exclude_patterns = [
    GENERATED_README,
    "dataset.md",
    "_hf_frontmatter.yaml",
    "_hf_model_card.md",
    "_hf_model_frontmatter.yaml",
]

html_theme = "furo"
html_logo = "assets/logo.svg"
html_title = "BlinkLinMulT"
html_static_path = ["assets"]
html_baseurl = "https://fodorad.github.io/BlinkLinMulT/"

html_theme_options = {
    # Matches the wordmark accent in assets/logo.svg.
    "light_css_variables": {"color-brand-primary": "#2f6fd0", "color-brand-content": "#2f6fd0"},
    "dark_css_variables": {"color-brand-primary": "#7bb0ff", "color-brand-content": "#7bb0ff"},
    "source_repository": "https://github.com/fodorad/BlinkLinMulT/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Warnings are errors in CI (`sphinx-build -W`), so an unresolved cross-reference
# fails the build rather than silently shipping a broken link.
nitpicky = False
