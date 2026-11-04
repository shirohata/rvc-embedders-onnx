"""Pinned RVC fairseq checkpoint -> Transformers layout, without importing fairseq.

This changes tensor names, not extraction semantics. Independent fairseq-runtime
parity is recorded separately; a successful layout conversion is not that proof.
"""
from __future__ import annotations

from pathlib import Path
import re

import torch
from transformers import HubertConfig


class DictionaryMetadata:
    """Inert checkpoint metadata; never import or construct fairseq objects."""


def fairseq_name(key: str) -> str:
    key = key.replace("masked_spec_embed", "mask_emb")
    key = re.sub(r"feature_extractor\.conv_layers\.(\d+)\.conv\.", r"feature_extractor.conv_layers.\1.0.", key)
    key = re.sub(r"feature_extractor\.conv_layers\.(\d+)\.layer_norm\.", r"feature_extractor.conv_layers.\1.2.", key)
    key = key.replace("feature_projection.projection.", "post_extract_proj.")
    key = key.replace("feature_projection.layer_norm.", "layer_norm.")
    key = key.replace("encoder.pos_conv_embed.conv.parametrizations.weight.original0", "encoder.pos_conv.0.weight_g")
    key = key.replace("encoder.pos_conv_embed.conv.parametrizations.weight.original1", "encoder.pos_conv.0.weight_v")
    key = key.replace("encoder.pos_conv_embed.conv.", "encoder.pos_conv.0.")
    key = key.replace(".attention.", ".self_attn.")
    key = re.sub(r"(encoder\.layers\.\d+)\.layer_norm\.", r"\1.self_attn_layer_norm.", key)
    key = key.replace(".feed_forward.intermediate_dense.", ".fc1.")
    return key.replace(".feed_forward.output_dense.", ".fc2.")


def load_checkpoint(path: Path):
    # The catalog caller verifies the pinned hash before this weights-only load.
    with torch.serialization.safe_globals([(DictionaryMetadata, "fairseq.data.dictionary.Dictionary")]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"]
    expected = dict(_name="hubert", extractor_mode="default", encoder_layers=12,
                    encoder_embed_dim=768, encoder_ffn_embed_dim=3072,
                    encoder_attention_heads=12, activation_fn="gelu", layer_norm_first=False,
                    conv_bias=False, final_dim=256, conv_pos=128, conv_pos_groups=16)
    if any(cfg["model"].get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported fairseq architecture; refusing implicit defaults")
    if cfg["task"].get("normalize") is not False or cfg["task"].get("sample_rate") != 16000:
        raise ValueError("Unsupported fairseq waveform preprocessing")
    if cfg["model"].get("conv_feature_layers") != "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2":
        raise ValueError("Unsupported fairseq convolution layout")
    config = HubertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                          intermediate_size=3072, classifier_proj_size=256,
                          conv_dim=[512] * 7, conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                          conv_stride=[5, 2, 2, 2, 2, 2, 2], conv_bias=False,
                          feat_extract_norm="group", feat_extract_activation="gelu",
                          do_stable_layer_norm=False, num_conv_pos_embeddings=128,
                          num_conv_pos_embedding_groups=16, layer_norm_eps=1e-5)
    return config, checkpoint["model"]


def map_state(state: dict, expected: dict) -> dict:
    converted = {}
    used = set()
    for name, template in expected.items():
        source_name = fairseq_name(name)
        if source_name not in state or state[source_name].shape != template.shape:
            raise ValueError(f"Missing or incompatible fairseq tensor: {source_name}")
        value = state[source_name]
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid fairseq tensor: {source_name}")
        converted[name] = value.float()
        used.add(source_name)
    # Classifier labels are unused by content-feature extraction. No other
    # unexpected source weights may silently disappear during conversion.
    unexpected = set(state) - used - {"label_embs_concat"}
    if unexpected:
        raise ValueError(f"Unexpected fairseq tensors: {sorted(unexpected)}")
    return converted
