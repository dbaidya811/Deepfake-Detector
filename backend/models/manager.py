import os
import json
import hashlib
import urllib.request
from typing import Dict, Tuple

MODELS_DIR = os.path.join(os.path.dirname(__file__))
MANIFEST_PATH = os.path.join(MODELS_DIR, 'model.json')
VERSIONS_PATH = os.path.join(MODELS_DIR, '.versions.json')

DEFAULT_TARGETS = {
    'video': 'video.onnx',
    'audio': 'audio.onnx',
}


def _read_json(path: str) -> Dict:
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _write_json(path: str, obj: Dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.download'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def check_and_update_models() -> Dict:
    """
    Reads MANIFEST_PATH if present:
    {
      "video": {"url": "https://.../video.onnx", "version": "v1", "sha256": "optional"},
      "audio": {"url": "https://.../audio.onnx", "version": "v1", "sha256": "optional"}
    }
    Downloads new versions to MODELS_DIR/{video.onnx,audio.onnx} and updates VERSIONS_PATH.
    Returns a dict with status per model.
    """
    manifest = _read_json(MANIFEST_PATH)
    versions = _read_json(VERSIONS_PATH)
    results: Dict[str, Dict] = {}

    for key, target_name in DEFAULT_TARGETS.items():
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            results[key] = {"status": "skipped", "reason": "no_manifest_entry"}
            continue
        url = entry.get('url')
        new_ver = entry.get('version')
        sha = entry.get('sha256')
        if not url or not new_ver:
            results[key] = {"status": "skipped", "reason": "missing_url_or_version"}
            continue

        current_ver = versions.get(key, {}).get('version')
        dest = os.path.join(MODELS_DIR, target_name)

        try:
            if current_ver != new_ver or not os.path.exists(dest):
                _download(url, dest)
                if sha:
                    file_hash = _hash_file(dest)
                    if file_hash.lower() != str(sha).lower():
                        raise RuntimeError(f"sha256 mismatch for {key}")
                versions[key] = {"version": new_ver, "path": dest}
                results[key] = {"status": "updated", "version": new_ver, "path": dest}
            else:
                results[key] = {"status": "current", "version": current_ver, "path": dest}
        except Exception as e:
            results[key] = {"status": "error", "error": str(e)}

    try:
        _write_json(VERSIONS_PATH, versions)
    except Exception:
        pass

    return {"results": results}
