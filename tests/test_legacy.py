import pytest
import torch

from vc_embedders.catalog import select_models
from vc_embedders.hubert import output_contract
from vc_embedders.legacy import fairseq_name, map_state


def test_default_scope_covers_both_official_paths_and_applio():
    selected = {item["id"] for item in select_models(["all"])}
    assert len(selected) == 8
    assert {"rvc-fairseq-base", "rvc-hubert-base", "applio-contentvec"} <= selected
    assert not any(name.startswith("wokada-") for name in selected)
    assert select_models(["wokada-contentvec"])[0]["id"] == "wokada-contentvec"
    assert output_contract("rvc-fairseq") == output_contract("rvc")


def test_legacy_mapping_preserves_every_feature_tensor_value():
    expected = {
        "feature_extractor.conv_layers.0.conv.weight": torch.zeros(2, 1, 3),
        "feature_extractor.conv_layers.0.layer_norm.weight": torch.zeros(2),
        "encoder.layers.0.attention.k_proj.weight": torch.zeros(2, 2),
        "encoder.layers.0.layer_norm.weight": torch.zeros(2),
        "encoder.layers.0.feed_forward.output_dense.weight": torch.zeros(2, 4),
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0": torch.zeros(1, 1, 3),
        "encoder.pos_conv_embed.conv.bias": torch.zeros(2),
        "final_proj.weight": torch.zeros(1, 2),
    }
    original = {fairseq_name(key): torch.arange(value.numel(), dtype=torch.float16).reshape(value.shape)
                for key, value in expected.items()}
    original["label_embs_concat"] = torch.ones(2, 2)
    mapped = map_state(original, expected)
    assert set(mapped) == set(expected)
    for key, value in mapped.items():
        assert value.dtype == torch.float32
        assert torch.equal(value, original[fairseq_name(key)].float())
    assert fairseq_name("encoder.layers.0.layer_norm.weight") == "encoder.layers.0.self_attn_layer_norm.weight"


@pytest.mark.parametrize("fault", ["missing", "shape", "unexpected", "nan"])
def test_legacy_conversion_rejects_incomplete_or_invalid_weights(fault):
    expected = {"final_proj.weight": torch.zeros(1, 2)}
    original = {"final_proj.weight": torch.ones(1, 2)}
    if fault == "missing": original.clear()
    if fault == "shape": original["final_proj.weight"] = torch.ones(2, 2)
    if fault == "unexpected": original["unmapped.weight"] = torch.ones(1)
    if fault == "nan": original["final_proj.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        map_state(original, expected)
