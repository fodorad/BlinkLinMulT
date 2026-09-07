"""The built wheel must contain what the code imports.

These are the tests a normal suite cannot provide: everything else runs against
the *source tree*, where every module is present by definition. A packaging
mistake only appears once the wheel is built and installed somewhere else, which
is exactly where it is most expensive to find.

That is not hypothetical. An earlier ``exclude = [..., "data/"]`` matched the
segment anywhere in a path, so hatchling silently dropped ``blinklinmult/data/``
along with the intended top-level ``data/``. Every test passed; the published
wheel raised ``ModuleNotFoundError`` on first import.

The build is slow, so these are opt-in::

    RUN_PACKAGING_TESTS=1 uv run python -m unittest tests.test_packaging -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

RUN_PACKAGING_TESTS = os.environ.get("RUN_PACKAGING_TESTS") == "1"
"""Whether to build a wheel. Off by default: a build takes tens of seconds."""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Repository root, where the build runs."""

REQUIRED_SUBPACKAGES = ("data", "train", "paper", "preprocess")
"""Subpackages the shipped code imports, so all must be in the wheel.

``preprocess`` is included deliberately. It was once excluded as "standalone
research CLIs", but :mod:`blinklinmult.pipeline` imports four of its modules,
and every heavy dependency it has (cv2, exordium, tqdm, h5py) is imported lazily
inside the functions that need them -- so shipping it costs an install nothing.
"""

FORBIDDEN_PREFIXES = ("data/raw", "data/processed")
"""Paths that must never reach a wheel: the corpora, which are gigabytes."""


@unittest.skipUnless(RUN_PACKAGING_TESTS, "set RUN_PACKAGING_TESTS=1 to build a wheel")
class TestWheelContents(unittest.TestCase):
    """What the built wheel does and does not contain."""

    wheel: Path
    names: list[str]

    @classmethod
    def setUpClass(cls) -> None:
        """Build one wheel and read its manifest, shared by every test here."""
        cls._directory = tempfile.TemporaryDirectory()
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["uv", "build", "--wheel", "-o", cls._directory.name],  # noqa: S607
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        built = sorted(Path(cls._directory.name).glob("*.whl"))
        cls.wheel = built[0]
        cls.names = zipfile.ZipFile(cls.wheel).namelist()

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary build directory."""
        cls._directory.cleanup()

    def test_every_imported_subpackage_ships(self) -> None:
        """Each subpackage the code imports is present, with modules in it.

        Checking for the directory alone would pass on an empty one, which is
        the shape the ``data/`` bug actually took.
        """
        for name in REQUIRED_SUBPACKAGES:
            with self.subTest(subpackage=name):
                modules = [
                    entry
                    for entry in self.names
                    if entry.startswith(f"blinklinmult/{name}/") and entry.endswith(".py")
                ]
                self.assertTrue(modules, f"blinklinmult/{name}/ is missing from the wheel")

    def test_corpora_are_not_shipped(self) -> None:
        """The built corpora stay out; they are gigabytes."""
        for entry in self.names:
            self.assertFalse(
                entry.startswith(FORBIDDEN_PREFIXES),
                f"{entry} should not be in the wheel",
            )

    def test_top_level_package_ships(self) -> None:
        """The public entry points are present."""
        for module in ("__init__.py", "detector.py", "registry.py", "pipeline.py"):
            with self.subTest(module=module):
                self.assertIn(f"blinklinmult/{module}", self.names)


@unittest.skipUnless(RUN_PACKAGING_TESTS, "set RUN_PACKAGING_TESTS=1 to build a wheel")
class TestInstalledWheelImports(unittest.TestCase):
    """The wheel must import from a clean environment, not just unzip cleanly.

    A manifest check catches a missing *file*; only an install catches a missing
    *dependency* or a module that imports something the wheel does not carry.
    """

    def test_imports_in_a_fresh_environment(self) -> None:
        """Install the wheel somewhere isolated and import the public surface."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(  # noqa: S603
                ["uv", "build", "--wheel", "-o", str(root)],  # noqa: S607
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
            )
            wheel = sorted(root.glob("*.whl"))[0]

            environment = root / "venv"
            subprocess.run(  # noqa: S603
                [
                    "uv",
                    "venv",
                    "-p",
                    f"{sys.version_info.major}.{sys.version_info.minor}",  # noqa: S607
                    str(environment),
                ],
                check=True,
                capture_output=True,
            )
            python = environment / "bin" / "python"
            subprocess.run(  # noqa: S603
                ["uv", "pip", "install", "--python", str(python), str(wheel)],  # noqa: S607
                check=True,
                capture_output=True,
            )

            probe = (
                "import blinklinmult, blinklinmult.pipeline, blinklinmult.overlay;"
                "from blinklinmult.preprocess.common import crop_square;"
                "from blinklinmult.data.schema import LEFT;"
                "assert hasattr(blinklinmult, 'BlinkDetector');"
                "print('ok')"
            )
            result = subprocess.run(  # noqa: S603
                [str(python), "-c", probe],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok", result.stdout)


class TestInferenceStaysLight(unittest.TestCase):
    """The ONNX path must not import torch.

    This is the whole reason the core dependency list is `numpy` + `pyyaml`:
    `pip install blinklinmult[onnx]` is ~100 MB against ~600 MB with the
    training stack, and torch alone is 469 MB of that.

    It has regressed before -- `detector.py` imported torch at module scope and
    used it purely as an array library, so a documented "no torch" Docker image
    was pulling it anyway. A subprocess is used because this test process has
    already imported half the project.
    """

    def _modules_after(self, code: str) -> set[str]:
        """Run code in a fresh interpreter and report the heavy modules loaded.

        Args:
            code (str): The snippet to run.

        Returns:
            set[str]: Which of the heavy packages ended up in ``sys.modules``.
        """
        probe = (
            "import sys\n"
            f"{code}\n"
            "print(','.join(m for m in ('torch','lightning','mlflow','pandas') "
            "if m in sys.modules))"
        )
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return {name for name in result.stdout.strip().split(",") if name}

    def test_importing_the_package_is_light(self) -> None:
        """`import blinklinmult` must pull nothing heavy."""
        self.assertEqual(self._modules_after("import blinklinmult"), set())

    def test_importing_the_detector_is_light(self) -> None:
        """The class itself must not drag in the training stack."""
        self.assertEqual(self._modules_after("from blinklinmult import BlinkDetector"), set())

    def test_scoring_an_onnx_model_never_touches_torch(self) -> None:
        """The load-bearing assertion: a graph scores without torch.

        Skipped when the local graph is absent, since it needs real weights.
        """
        graph = Path("artifacts/onnx/blinkcnn.onnx")
        if not graph.is_file():
            self.skipTest("run `make export-blinkcnn-onnx` first")
        loaded = self._modules_after(
            "import numpy as np\n"
            "from blinklinmult import BlinkDetector\n"
            f"d = BlinkDetector.from_pretrained('blinkcnn-onnx', weights={str(graph)!r})\n"
            "d.score(np.zeros((4, 3, 64, 64), dtype='float32'))"
        )
        self.assertEqual(loaded, set(), f"the ONNX path pulled {loaded}")

    def test_the_core_dependency_list_stays_small(self) -> None:
        """A heavy package added to `dependencies` defeats the whole design."""
        import tomllib

        with Path("pyproject.toml").open("rb") as handle:
            core = tomllib.load(handle)["project"]["dependencies"]
        names = {entry.split(">")[0].split("[")[0].strip() for entry in core}
        heavy = names & {"torch", "lightning", "torchmetrics", "mlflow", "pandas", "timm"}
        self.assertEqual(heavy, set(), f"{heavy} belongs in an extra, not in core")


class TestGateParity(unittest.TestCase):
    """CI and `make check` must install the same extras.

    They drifted once: the Makefile gained `serve` while CI did not, and because
    the affected tests *skip* rather than fail when their imports are missing,
    the difference was invisible -- fifteen service tests had stopped running
    and nothing said so.
    """

    @staticmethod
    def _extras(text: str, marker: str) -> set[str]:
        """Pull an extras list out of a config file.

        Args:
            text (str): The file's contents.
            marker (str): A substring on the line holding the extras.

        Returns:
            set[str]: The extra names found.
        """
        import re

        for line in text.splitlines():
            if marker in line:
                inside = re.search(r"\[([a-z,\s]+)\]", line)
                if inside:
                    return {name.strip() for name in inside.group(1).split(",") if name.strip()}
        return set()

    def test_ci_installs_what_the_gate_installs(self) -> None:
        """A skipped test is not a passing test."""
        makefile = Path("Makefile").read_text()
        workflow = Path(".github/workflows/ci.yml").read_text()

        gate = set()
        for line in makefile.splitlines():
            if line.startswith("EXTRAS :="):
                gate = {part for part in line.split() if not part.startswith("--")}
                gate -= {"EXTRAS", ":="}
                break

        ci = self._extras(workflow, "uv pip install --system -e")
        missing = gate - ci - {"dev"}
        self.assertEqual(
            missing, set(), f"CI does not install {missing}, so those tests skip there"
        )


if __name__ == "__main__":
    unittest.main()
