import json
import os
import re
from typing import Any, Dict, List


PATTERNS = [
    (re.compile(r"\bfrom\s+aimemory\b"), "from arc1"),
    (re.compile(r"\bimport\s+aimemory\b"), "import arc1"),
    (re.compile(r"\baimemory\s+"), "arc1 "),
    (re.compile(r"\baimemory-engine\b"), "arc1-engine"),
]


def _candidate(path: str) -> bool:
    p = path.lower()
    return p.endswith((".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".sh"))


def build_migration_report(root: str, rewrite: bool = False) -> Dict[str, Any]:
    root = os.path.abspath(root or ".")
    hits: List[Dict[str, Any]] = []
    scanned = 0
    changed = 0
    for d, _, files in os.walk(root):
        if ".git" in d.split(os.sep):
            continue
        for fn in files:
            fp = os.path.join(d, fn)
            if not _candidate(fp):
                continue
            scanned += 1
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
            except Exception:
                continue
            new = raw
            file_hits = 0
            for pat, rep in PATTERNS:
                nnew, n = pat.subn(rep, new)
                if n > 0:
                    new = nnew
                    file_hits += int(n)
            if file_hits > 0:
                rel = os.path.relpath(fp, root)
                hits.append({"file": rel, "matches": int(file_hits)})
                if rewrite and new != raw:
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(new)
                    changed += 1
    return {
        "root": root,
        "scanned_files": int(scanned),
        "files_with_hits": int(len(hits)),
        "files_rewritten": int(changed),
        "hits": hits[:500],
        "rewrite": bool(rewrite),
    }


def write_migration_report(root: str, out_path: str, rewrite: bool = False) -> Dict[str, Any]:
    rep = build_migration_report(root=root, rewrite=rewrite)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    return rep
