import os

import pytest
from jinja2 import Template

# Ensure template path is set to test location
TEMPLATE_PATH = "tests/assets/test_prompt_template.j2"

# Sample template used in tests
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


class MockTokenizer:
    def __call__(
        self,
        prompt,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
    ):
        return MockBatchEncoding({"input_ids": [[1, 2, 3]]})

    def decode(self, ids, skip_special_tokens=False):
        return self._mock_output

    def batch_decode(self, batch_ids, skip_special_tokens=False):
        return [self._mock_output for _ in batch_ids]

    @property
    def eos_token_id(self):
        return 0


class MockModel:
    def generate(self, *args, **kwargs):
        return [0, 1, 2, 3]

    @property
    def device(self):
        return "cpu"


@pytest.fixture(scope="session", autouse=True)
def create_test_template():
    os.makedirs("tests/assets", exist_ok=True)
    with open(TEMPLATE_PATH, "w") as f:
        f.write(SAMPLE_TEMPLATE)
    os.environ["TEMPLATE_PATH"] = TEMPLATE_PATH


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

    # Clear cached template to pick up test template
    llm_few_shot.get_template.cache_clear()

    tokenizer = MockTokenizer()
    tokenizer._mock_output = mock_output
    model = MockModel()

    results = llm_few_shot.classify_post(["Test post"], tokenizer, model)
    label, output = results[0]

    assert label == expected_label
    assert mock_output in output
