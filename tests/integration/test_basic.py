"""Basic test to verify CI is working"""


def test_basic():
    """Basic test that always passes"""
    assert True


def test_math():
    """Simple math test"""
    assert 1 + 1 == 2
    assert 2 * 3 == 6


def test_string():
    """Simple string test"""
    assert "hello" + " world" == "hello world"
    assert len("test") == 4
