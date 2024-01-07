import pytest
from all_clip import load_clip


@pytest.mark.parametrize("message", ["hello", "world"])
def test_load_clip(message):
    print("tmp")
