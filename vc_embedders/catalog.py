from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


def read_catalog() -> dict:
    return json.loads(Path(__file__).with_name("catalog.json").read_text(encoding="utf-8-sig"))


def select_models(names: list[str]) -> list[dict]:
    catalog = read_catalog()
    models = catalog["models"]
    if names == ["all"]:
        deferred = set(catalog.get("deferred_models", []))
        return [item for item in models if item["id"] not in deferred]
    by_id = {item["id"]: item for item in models}
    unknown = set(names) - by_id.keys()
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return [by_id[name] for name in dict.fromkeys(names)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_file(path: Path, spec: dict) -> bool:
    return path.is_file() and path.stat().st_size == spec["size"] and sha256(path) == spec["sha256"]


def source_url(model: dict, spec: dict) -> str:
    from urllib.parse import quote

    if "url" in spec:
        if not spec["url"].startswith("https://"):
            raise ValueError("Pinned source URL must use HTTPS")
        return spec["url"]
    return f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{quote(spec["path"], safe="/")}'


def fetch(model: dict, cache: Path, local_dir: Path | None = None) -> Path:
    """Even existing/local weights must match the pinned upstream bytes before loading."""
    directory = cache / model.get("cache_id", model["id"])
    directory.mkdir(parents=True, exist_ok=True)
    for spec in model["files"]:
        target = directory / spec["name"]
        if verified_file(target, spec):
            continue
        local = local_dir / spec["name"] if local_dir else None
        if local and not verified_file(local, spec):
            raise ValueError(f"Local source does not match pinned SHA-256/size: {local}")
        descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".partial")
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            if local:
                shutil.copyfile(local, temporary)
            else:
                print(f'Fetching {model["id"]}/{spec["name"]}', flush=True)
                request = urllib.request.Request(source_url(model, spec), headers={"User-Agent": "vc-embedders/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if not verified_file(temporary, spec):
                raise ValueError(f'SHA-256/size mismatch: {model["id"]}/{spec["name"]}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return directory


def check_sources(model: dict, cache: Path) -> Path:
    directory = cache / model.get("cache_id", model["id"])
    for spec in model["files"]:
        if not verified_file(directory / spec["name"], spec):
            raise ValueError(f'Fetch first: pinned source missing or corrupt: {model["id"]}/{spec["name"]}')
    return directory
