from unittest.mock import MagicMock

import pytest
from jinja2 import Template

# 공통 템플릿
SAMPLE_TEMPLATE = """<s> [INST] {{ instruction }}
A post should be classified as "1" (Yes) if it includes:
{{ yes_criteria }}
A post should be classified as "0" (No) if it:
{{ no_criteria }}
Now decide:
Post: "{{ post }}"
[/INST]
Output:"""


class MockBatchEncoding(dict):
    def to(self, device):
        return self


# Common mock tokenizer
class MockTokenizer:
    def __call__(self, prompt, return_tensors="pt"):
        return MockBatchEncoding({"input_ids": [[1, 2, 3]]})

    def decode(self, ids, skip_special_tokens=False):
        return self._mock_output

    @property
    def eos_token_id(self):
        return 0


# 공통 mock model
class MockModel:
    def generate(self, *args, **kwargs):
        return [[0, 1, 2, 3]]

    @property
    def device(self):
        return "cpu"


@pytest.mark.parametrize(
    "mock_output,expected_label",
    [
        ("</INST>\nOutput: 1", "Yes"),
        ("</INST>\nOutput: 0", "No"),
        ("</INST>\nOutput: 🤷", "Uncertain"),
    ],
)
def test_classify_post_variants(mock_output, expected_label):
    from processing import llm_few_shot

    # 템플릿 재정의
    llm_few_shot.template = Template(SAMPLE_TEMPLATE)

    # tokenizer/model 생성 및 mock 설정
    tokenizer = MockTokenizer()
    tokenizer._mock_output = mock_output
    model = MockModel()

    label, output = llm_few_shot.classify_post("Test post", tokenizer, model)

    assert label == expected_label
    assert mock_output in output
