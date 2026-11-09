import warnings

import pytest
import torch

from surge.models.modules.heads.nad import NAD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_nad_bf16_warning(compiled):
    model = NAD(dim_in=[64], dim_out=[3], embed_dim=[64], depth=1).cuda().eval()
    if compiled:
        model = torch.compile(model, fullgraph=True)
    features = [torch.randn(1, 64, 9, 11, device="cuda")]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with pytest.warns(UserWarning, match="BF16 autocast.*FP16 is recommended"):
            model(features)
        with (
            torch.compiler.set_stance("fail_on_recompile"),
            pytest.warns(UserWarning, match="BF16 autocast.*FP16 is recommended"),
        ):
            model(features)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="NAD is running under BF16")
        for dtype in (torch.float16, torch.float32):
            with torch.autocast("cuda", dtype=dtype):
                model(features)

        features = [features[0].bfloat16()]
        model(features)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.autocast("cuda", dtype=torch.float16):
                model(features)
