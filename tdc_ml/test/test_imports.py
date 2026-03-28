# -*- coding: utf-8 -*-

from __future__ import print_function

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestImports(unittest.TestCase):

    def test_metadata_does_not_require_pkg_resources(self):
        repo_root = Path(__file__).resolve().parents[2]
        code = textwrap.dedent(
            """
            import builtins
            import runpy
            from pathlib import Path

            repo_root = Path.cwd()
            real_import = builtins.__import__

            def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "pkg_resources":
                    raise ModuleNotFoundError("pkg_resources is blocked by regression test")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = blocked_import
            runpy.run_path(str(repo_root / "tdc_ml" / "metadata.py"))
            """
        )

        subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
