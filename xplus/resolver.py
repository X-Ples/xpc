from __future__ import annotations

import os
import sys
import json
import hashlib
import urllib.request
import urllib.error

from . import REGISTRY_BASE, VERSION

HOME_LIBS = os.path.expanduser("~/.xpc/libs")
BUNDLED_LIBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs")
LOCKFILE = "xpc.lock"


def ensure_dirs():
    os.makedirs(HOME_LIBS, exist_ok=True)


def module_name(path: str) -> str:
    if path.endswith(".xp"):
        return path[:-3]
    return path


def module_path(mod: str) -> str:
    return mod.replace("::", "/") + ".xp"

def parse_mini_toml(text: str) -> dict:
    root = {}
    cur = root
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            cur = root
            for part in name.split("."):
                cur = cur.setdefault(part.strip(), {})
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            elif v in ("true", "false"):
                v = (v == "true")
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            cur[k] = v
    return root


def load_registry() -> dict:
    reg_path = os.path.join(BUNDLED_LIBS, "registry.toml")
    if os.path.exists(reg_path):
        with open(reg_path, encoding="utf-8") as f:
            return parse_mini_toml(f.read())
    return {}


def fetch_live_registry(timeout: float = 5.0) -> dict:
    url = REGISTRY_BASE + "registry.toml"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"xpc/{VERSION}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return parse_mini_toml(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return {}


def get_registry(live: bool = True) -> dict:
    merged = fetch_live_registry() if live else {}
    bundled = load_registry()
    for section, values in bundled.items():
        if isinstance(values, dict):
            dst = merged.setdefault(section, {})
            for name, info in values.items():
                dst.setdefault(name, info)
    return merged

class ResolveResult:
    def __init__(self, mod, source, origin, file):
        self.mod = mod
        self.source = source
        self.origin = origin
        self.file = file


def resolve_module(mod: str, project_dir: str = ".", silent=True) -> ResolveResult:
    fname = module_path(mod)
    local = os.path.join(project_dir, fname)
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            return ResolveResult(mod, f.read(), "local", local)
    base_local = os.path.join(project_dir, os.path.basename(fname))
    if os.path.exists(base_local):
        with open(base_local, encoding="utf-8") as f:
            return ResolveResult(mod, f.read(), "local", base_local)
    cache = os.path.join(HOME_LIBS, os.path.basename(fname))
    if os.path.exists(cache) and not os.path.exists(local):
        with open(cache, encoding="utf-8") as f:
            return ResolveResult(mod, f.read(), "cache", cache)
    bundled = os.path.join(BUNDLED_LIBS, os.path.basename(fname))
    if os.path.exists(bundled):
        with open(bundled, encoding="utf-8") as f:
            return ResolveResult(mod, f.read(), "bundled", bundled)
    url = REGISTRY_BASE + fname
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"xpc/{VERSION}"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            src = resp.read().decode("utf-8")
        ensure_dirs()
        with open(cache, "w", encoding="utf-8") as f:
            f.write(src)
        return ResolveResult(mod, src, "github", cache)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        if not silent:
            raise
    return None

def pkg_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def install_packages(names, no_cache=False, out=sys.stdout) -> list:
    ensure_dirs()
    registry = get_registry()
    packages = registry.get("packages", {})
    installed = []
    for name in names:
        info = packages.get(name, {})
        version = info.get("version", "1.0.0")
        author = info.get("author", "")
        origin = None
        bundled = os.path.join(BUNDLED_LIBS, f"{name}.xp")
        if os.path.exists(bundled):
            with open(bundled, encoding="utf-8") as f:
                src = f.read()
            with open(os.path.join(HOME_LIBS, f"{name}.xp"), "w", encoding="utf-8") as f:
                f.write(src)
            origin = "bundled-registry"
        else:
            try:
                url = REGISTRY_BASE + f"{name}/{name}.xp"
                req = urllib.request.Request(url, headers={"User-Agent": f"xpc/{VERSION}"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    src = resp.read().decode("utf-8")
                with open(os.path.join(HOME_LIBS, f"{name}.xp"), "w", encoding="utf-8") as f:
                    f.write(src)
                origin = "github"
            except Exception:
                origin = None
        installed.append((name, origin, version, src if origin else None, author))
    return installed


def list_installed() -> list:
    ensure_dirs()
    out = []
    if os.path.isdir(HOME_LIBS):
        for fn in sorted(os.listdir(HOME_LIBS)):
            if fn.endswith(".xp"):
                path = os.path.join(HOME_LIBS, fn)
                size = os.path.getsize(path)
                out.append((fn[:-3], path, size))
    return out


def uninstall_packages(names) -> list:
    removed = []
    for name in names:
        p = os.path.join(HOME_LIBS, f"{name}.xp")
        if os.path.exists(p):
            os.remove(p)
            removed.append(name)
    return removed


def search_registry(keyword) -> list:
    registry = get_registry()
    packages = registry.get("packages", {})
    res = []
    for name, info in packages.items():
        if keyword.lower() in name.lower() or keyword.lower() in str(info.get("description", "")).lower():
            res.append((name, info))
    return res


def info_package(name):
    registry = get_registry()
    packages = registry.get("packages", {})
    return packages.get(name)


def cache_size_bytes() -> int:
    total = 0
    for root in (HOME_LIBS, "/tmp/xpc-builds"):
        if os.path.isdir(root):
            for fn in os.listdir(root):
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    return total


def write_lockfile(project_dir=".", entries=None):
    lock = {"lockfile_version": 1, "generated_by": f"xpc {VERSION}", "packages": {}}
    for name, origin, version, src, _author in (entries or []):
        if src:
            lock["packages"][name] = {
                "version": version,
                "origin": origin,
                "sha256": hashlib.sha256(src.encode()).hexdigest(),
            }
    path = os.path.join(project_dir, LOCKFILE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# xpc.lock — generated by xpc {VERSION}\n")
        f.write(f'[meta]\nlockfile_version = 1\ngenerated_by = "xpc {VERSION}"\n\n')
        for name, info in lock["packages"].items():
            f.write(f"[packages.{name}]\n")
            f.write(f'version = "{info["version"]}"\n')
            f.write(f'origin = "{info["origin"]}"\n')
            f.write(f'sha256 = "{info["sha256"]}"\n\n')
    return path