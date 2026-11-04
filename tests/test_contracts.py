import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from transformers import HubertConfig

from vc_embedders.catalog import fetch, read_catalog, verified_file
from vc_embedders.hubert import ExportWrapper, HubertWithProjection, load_model, output_contract
from vc_embedders.validation import compare_arrays, validate


def test_registry_pins_every_file_and_revision():
    catalog = read_catalog()
    assert len({m["id"] for m in catalog["models"]}) == 10
    for model in catalog["models"]:
        assert len(model["revision"]) == 40
        for file in model["files"]:
            assert len(file["sha256"]) == 64
            assert file["size"] > 0
            assert "/" not in file["name"] and "\\" not in file["name"]


def test_corrupt_cached_or_local_weights_are_rejected(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    path = local / "pytorch_model.bin"
    path.write_bytes(b"untrusted")
    spec = dict(name=path.name, size=9, sha256=hashlib.sha256(b"trusted!!").hexdigest())
    item = dict(id="example", files=[spec])
    assert not verified_file(path, spec)
    with pytest.raises(ValueError, match="does not match"):
        fetch(item, tmp_path / "cache", local)
    assert not list((tmp_path / "cache").rglob("*.partial"))


def test_official_and_applio_v1_are_different_layers():
    official = output_contract("rvc")[0]
    applio = output_contract("applio", True)[1]
    assert official["layer"] == 9 and official["name"] == "unit9"
    assert applio["layer"] == 12 and applio["name"] == "unit12_proj"
    assert all(x["channels"] == 768 for x in output_contract("applio"))


def tiny_model():
    config = HubertConfig(hidden_size=12, num_hidden_layers=12, num_attention_heads=2,
                          intermediate_size=16, conv_dim=(8, 8, 8, 8, 8, 8, 8),
                          conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                          num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                          classifier_proj_size=4, mask_time_prob=0.0, layerdrop=0.0)
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    return HubertWithProjection(config).eval()


def test_missing_projection_must_not_be_exported_as_trained(tmp_path):
    model = tiny_model()
    (tmp_path / "config.json").write_text(model.config.to_json_string())
    state = {key: value for key, value in model.state_dict().items() if not key.startswith("final_proj.")}
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="no trained final_proj"):
        load_model(tmp_path, require_projection=True)
    assert load_model(tmp_path, require_projection=False).config.hidden_size == 12


def test_dynamic_onnx_preserves_features_at_untraced_lengths(tmp_path):
    model = tiny_model()
    outputs = copy.deepcopy(output_contract("rvc"))
    outputs[0]["channels"] = 4
    outputs[1]["channels"] = 12
    wrapper = ExportWrapper(model, outputs).eval()
    path = tmp_path / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(wrapper, (torch.randn(1, 1600),), str(path),
                          input_names=["audio"], output_names=["unit9", "unit12"],
                          dynamic_axes={"audio": {1: "samples"}, "unit9": {1: "frames"}, "unit12": {1: "frames"}},
                          opset_version=17, dynamo=False)
    report = validate(path, model, outputs, [1600, 3201], [], 1, 1e-4, 1e-4)
    assert report["passed"]
    assert len(report["cases"]) == 12
    assert len({row["shape"][1] for row in report["cases"]}) == 2


def test_validation_rejects_shape_nan_and_wrong_values():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_arrays(np.zeros((1, 2)), np.zeros((1, 3)), atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="Non-finite"):
        compare_arrays(np.array([np.nan]), np.array([0.0]), atol=1e-4, rtol=1e-4)
    assert not compare_arrays(np.ones(4), np.zeros(4), atol=1e-4, rtol=1e-4)["passed"]
    assert not compare_arrays(np.full(4, 5e-4), np.zeros(4), atol=1e-3, rtol=1e-4)["passed"]
