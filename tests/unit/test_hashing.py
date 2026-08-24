from processing.hashing import sha256_json, sha256_text


def test_json_hash_is_key_order_independent():
    left = sha256_json({"b": 1, "a": {"y": 2, "x": 3}})
    right = sha256_json({"a": {"x": 3, "y": 2}, "b": 1})
    assert left == right
    assert len(left) == 64


def test_text_hash_is_stable():
    assert sha256_text("synthetic") == sha256_text("synthetic")
    assert sha256_text("synthetic") != sha256_text("Synthetic")
