import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("ctgan") is None, reason="optional CTGAN dependency not installed")
def test_ctgan_importable_when_optional_dependency_installed():
    from ctgan import CTGAN

    assert CTGAN is not None
