import pytest
import torch

from surge import SurGe


@pytest.mark.parametrize("model_name", ["karimknaebel/surge-large"])
def test_pretrained(model_name: str):
    SurGe.from_pretrained(model_name, strict=True)


@pytest.mark.parametrize(
    "model_name,expected",
    [
        (
            "karimknaebel/surge-large",
            [-0.429520845413208, -1.8300488591194153e-07, 1.074447751045227],
        )
    ],
)
def test_infer(model_name: str, expected: list[float]):
    model = SurGe.from_pretrained(model_name, strict=True)

    rng = torch.Generator()
    rng.manual_seed(0)
    image = torch.randn(1, 3, 144, 144, generator=rng)

    out = model.infer(image, num_tokens=81)
    actual = torch.quantile(
        out["points"][0, 2].reshape(-1), torch.tensor([0.25, 0.5, 0.75])
    )

    # torch.testing.assert_close(actual, torch.tensor(expected))
    assert actual.tolist() == expected
