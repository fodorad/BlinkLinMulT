"""Run the suite and exit without C-level finalisation.

``python -m tests`` instead of ``python -m unittest discover``. The two run
exactly the same tests; the difference is the exit path.

**Why this exists.** Roughly one run in twelve ended with

    Ran 1796 tests ... OK
    libc++abi: terminating due to uncaught exception of type
    std::__1::system_error: recursive_mutex lock failed: Invalid argument

and a 134 exit code -- *after* every test had passed. There is no Python frame
in it: the abort happens while CPython finalises native extension modules, and
this process has several loaded at once (onnxruntime sessions, mediapipe's GL
context, torch's thread pools). Their destructors race each other, and one of
them takes a lock whose owning thread is already gone.

Nothing in this repository can order those destructors. What it *can* do is not
run them: once the results are printed and the streams are flushed, the process
has no work left, so :func:`os._exit` is the honest ending rather than a
workaround. Verified by measurement -- 0 failures in 12 runs against 1 in 12
through the normal exit path.

The cost is that ``atexit`` hooks do not fire. Two consequences are handled
here: :func:`blinklinmult.stream._join_live_scorers` is not relied on for
correctness (a scorer must still be stopped by its owner, and every test does),
and **coverage is saved explicitly** -- ``coverage`` normally writes its data
from an ``atexit`` hook, so skipping that dropped a threaded module from 92% to
32% before this was added.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

TOP_LEVEL = Path(__file__).resolve().parents[1]
"""The repository root.

Discovery must start here, not at ``tests/``: modules such as
``demos.docker.serve`` and ``tools.*`` are imported by their tests, and a
``tests``-rooted ``sys.path`` makes those imports fail into a *skip* rather than
an error. That silently hid fifteen service tests once already.
"""


def main() -> int:
    """Discover and run every test.

    Returns:
        int: ``0`` when the suite passed, ``1`` otherwise.
    """
    suite = unittest.TestLoader().discover("tests", top_level_dir=str(TOP_LEVEL))
    verbosity = 2 if "-v" in sys.argv else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


def _save_coverage() -> None:
    """Flush coverage data before the process exits abruptly.

    ``coverage`` hooks ``atexit`` to write ``.coverage``; :func:`os._exit` skips
    that, so the run's data would be lost -- silently, as a plausible-looking
    lower number rather than an error.
    """
    try:
        import coverage
    except ImportError:
        return
    current = coverage.Coverage.current()
    if current is not None:
        current.stop()
        current.save()


if __name__ == "__main__":
    code = main()
    _save_coverage()
    sys.stdout.flush()
    sys.stderr.flush()
    # See the module docstring: skip C-level finalisation deliberately.
    os._exit(code)
