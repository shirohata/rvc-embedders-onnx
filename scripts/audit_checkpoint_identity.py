"""Compare checkpoint tensors without importing or executing checkpoint classes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch


class DictionaryMetadata:
    pass


class LoggerMetadata:
    pass


def load(path):
    # These inert stand-ins only hold metadata state; neither imports fairseq nor
    # initializes a logger. All other nonstandard pickle globals stay rejected.
    with torch.serialization.safe_globals([
        (DictionaryMetadata, "fairseq.data.dictionary.Dictionary"),
        (LoggerMetadata, "pytorch_lightning.loggers.wandb.WandbLogger"),
    ]):
        return torch.load(path, map_location="cpu", weights_only=True)


def original_name(key):
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
    key = key.replace(".feed_forward.output_dense.", ".fc2.")
    return key


def compare(source, candidate, spin=False):
    if spin:
        source = {**{k.removeprefix("encoder.model."): v for k,v in source.items() if k.startswith("encoder.model.")},
                  **{k.replace("pred_head.layers.0.", "final_proj."): v for k,v in source.items() if k.startswith("pred_head.layers.0.")}}
    rows = []
    for key, value in candidate.items():
        mapped = original_name(key)
        if mapped not in source:
            rows.append(dict(name=key, original_name=mapped, result="missing"))
            continue
        original = source[mapped]
        if list(original.shape) != list(value.shape):
            rows.append(dict(name=key, original_name=mapped, result="shape-mismatch"))
            continue
        original = original.float()
        value = value.float()
        exact = torch.equal(original, value)
        fp16 = torch.equal(original.half().float(), value)
        rows.append(dict(name=key, original_name=mapped,
                         result="exact" if exact else "fp16-rounded" if fp16 else "different",
                         max_abs=float((original-value).abs().max())))
    counts={name:sum(r['result']==name for r in rows) for name in ['exact','fp16-rounded','different','missing','shape-mismatch']}
    return dict(candidate_tensors=len(candidate), counts=counts,
                all_exact=counts['exact']==len(candidate),
                all_exact_or_fp16_rounded=counts['exact']+counts['fp16-rounded']==len(candidate), tensors=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True,help='Pinned source checksum; checked before parsing')
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--spin',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    with args.source.open('rb') as stream:
        source_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    if source_sha256 != args.source_sha256:
        raise ValueError('Source SHA-256 does not match the pinned checkpoint')
    torch.set_num_threads(4)
    original=load(args.source)
    original=original.get('model',original.get('state_dict',original))
    candidate=load(args.candidate)
    report=compare(original,candidate,args.spin)
    report.update(source_sha256=source_sha256,
                  candidate_sha256=hashlib.file_digest(args.candidate.open('rb'),'sha256').hexdigest(),torch=torch.__version__)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tensors'}),flush=True)


if __name__=='__main__':main()
