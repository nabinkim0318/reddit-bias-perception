## utils/tokenize.py


def batch_tokenize(texts, tokenizer, max_length=512):
    if not texts:
        texts = [""]
    texts = [str(t or "") for t in texts]
    return tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
