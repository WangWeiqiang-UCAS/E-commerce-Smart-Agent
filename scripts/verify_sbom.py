"""Verify the checked-in CycloneDX SBOM against the repository manifests.

This deliberately uses only the standard library so CI can verify supply-chain
metadata before installing project dependencies.
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"SBOM verification failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def component_versions(sbom: dict) -> dict[str, str]:
    if sbom.get("bomFormat") != "CycloneDX":
        fail("bomFormat must be CycloneDX")
    if sbom.get("specVersion") != "1.5":
        fail("specVersion must be 1.5")
    components = sbom.get("components")
    if not isinstance(components, list):
        fail("components must be an array")
    versions: dict[str, str] = {}
    for component in components:
        if not isinstance(component, dict):
            fail("each component must be an object")
        name, version = component.get("name"), component.get("version")
        licenses = component.get("licenses")
        if not isinstance(name, str) or not isinstance(version, str):
            fail("every component requires name and version")
        if not isinstance(licenses, list) or not licenses:
            fail(f"{name} has no license declaration")
        versions[name] = version
    return versions


def expect(versions: dict[str, str], name: str, version: str) -> None:
    actual = versions.get(name)
    if actual != version:
        fail(f"{name} is {actual!r} in SBOM, expected {version!r}")


def poetry_version(specification: str) -> str:
    """The SBOM stores the pinned base version for a Poetry caret spec."""
    return specification.lstrip("^~=>< ")


def main() -> None:
    sbom = json.loads((ROOT / "third_party" / "SBOM").read_text(encoding="utf-8"))
    versions = component_versions(sbom)

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["tool"]["poetry"]["dependencies"]
    for dependency, sbom_name in {
        "fastapi": "fastapi",
        "langgraph": "langgraph",
        "langchain": "langchain",
        "sqlmodel": "sqlmodel",
        "asyncpg": "asyncpg",
        "pgvector": "pgvector",
        "redis": "redis-py",
        "kombu": "kombu",
        "httpx": "httpx",
        "alembic": "alembic",
        "pyjwt": "pyjwt",
        "pypdf": "pypdf",
        "tenacity": "tenacity",
    }.items():
        expect(versions, sbom_name, poetry_version(dependencies[dependency]))

    for app in ("agent-bff-ts", "agent-desk-ts"):
        package_lock = json.loads((ROOT / "apps" / app / "package-lock.json").read_text(encoding="utf-8"))
        packages = package_lock.get("packages", {})
        for package_name, sbom_name in {
            "node_modules/next": "next",
            "node_modules/react": "react",
            "node_modules/typescript": "typescript",
        }.items():
            package = packages.get(package_name, {})
            if not isinstance(package.get("version"), str):
                fail(f"{app} lockfile has no version for {package_name}")
            expect(versions, sbom_name, package["version"])

    upstream = (ROOT / "third_party" / "UPSTREAMS.md").read_text(encoding="utf-8")
    revision = re.search(r"Imported revision: `([0-9a-f]+)`", upstream)
    if not revision:
        fail("macrozheng/mall imported revision is missing")
    expect(versions, "macrozheng/mall", revision.group(1)[:8])
    print("SBOM verification passed")


if __name__ == "__main__":
    main()
