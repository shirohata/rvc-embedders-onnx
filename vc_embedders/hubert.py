from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import HubertConfig, HubertModel


class HubertWithProjection(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_model(directory: Path, require_projection: bool, source_kind: str = "transformers") -> HubertWithProjection:
    if source_kind == "fairseq":
        from .legacy import load_checkpoint
        config, legacy_state = load_checkpoint(directory / "hubert_base.pt")
    else:
        config = HubertConfig.from_dict(json.loads((directory / "config.json").read_text(encoding="utf-8")))
    # Eager attention exports to ordinary ONNX ops and avoids SDPA backend changes.
    config._attn_implementation = "eager"
    model = HubertWithProjection(config)
    if source_kind == "fairseq":
        from .legacy import map_state
        state = map_state(legacy_state, model.state_dict())
    else:
        state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if require_projection and not {"final_proj.weight", "final_proj.bias"} <= state.keys():
        raise ValueError("Checkpoint has no trained final_proj; refusing a random 256-channel projection")
    mismatch = model.load_state_dict(state, strict=False)
    allowed_missing = set() if require_projection else {"final_proj.weight", "final_proj.bias"}
    if set(mismatch.missing_keys) - allowed_missing or mismatch.unexpected_keys:
        raise ValueError(f"Checkpoint architecture mismatch: {mismatch}")
    return model.float().eval()


def output_contract(profile: str, include_applio_v1: bool = False) -> list[dict]:
    if profile in {"rvc", "rvc-fairseq"}:
        if include_applio_v1:
            raise ValueError("--include-applio-v1 is only valid for Applio")
        return [dict(name="unit9", layer=9, projection=True, channels=256),
                dict(name="unit12", layer=12, projection=False, channels=768)]
    if profile != "applio":
        raise ValueError(f"Unknown extraction profile: {profile}")
    outputs = [dict(name="unit12", layer=12, projection=False, channels=768)]
    if include_applio_v1:
        # Applio's legacy v1 path projects layer 12, unlike official RVC v1.
        # Do not name this unit9: vc-rs prefers that name for official 256-D models.
        outputs.append(dict(name="unit12_proj", layer=12, projection=True, channels=256))
    return outputs


class ExportWrapper(nn.Module):
    def __init__(self, model: HubertWithProjection, outputs: list[dict]):
        super().__init__()
        self.model = model
        self.outputs = outputs

    def forward(self, audio):
        result = self.model(audio, attention_mask=None, output_hidden_states=True, return_dict=True)
        values = []
        for output in self.outputs:
            value = result.hidden_states[output["layer"]]
            values.append(self.model.final_proj(value) if output["projection"] else value)
        return tuple(values)


def reference_features(model: HubertWithProjection, audio, outputs: list[dict]):
    """Direct source-app formulas, kept separate from the ONNX export wrapper."""
    result = model(audio, attention_mask=None, output_hidden_states=True, return_dict=True)
    return {item["name"]: (model.final_proj(result.hidden_states[item["layer"]])
                          if item["projection"] else result.hidden_states[item["layer"]])
            for item in outputs}
