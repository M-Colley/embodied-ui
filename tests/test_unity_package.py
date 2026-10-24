"""Offline checks of the Unity package's layout."""

import json
import re
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "unity" / "io.github.m-colley.embodied-ui"


def test_components_live_in_files_named_after_them():
    # Unity attaches a component in the Editor, and restores it from a saved
    # scene, only when its class and file names match.
    for path in (PACKAGE / "Runtime").rglob("*.cs"):
        text = path.read_text(encoding="utf-8")
        components = re.findall(
            r"public (?:sealed |abstract )?class (\w+) : (?:MonoBehaviour|Embodied\w+)",
            text,
        )
        assert components in ([], [path.stem]), (path.name, components)


def test_every_asset_has_a_meta_file_with_a_unique_guid():
    guids = {}
    for path in PACKAGE.rglob("*"):
        relative = path.relative_to(PACKAGE)
        if path.suffix == ".meta" or any(p.endswith("~") for p in relative.parts):
            continue
        meta = Path(f"{path}.meta")
        assert meta.exists(), relative
        guid = re.search(r"^guid: (\w+)$", meta.read_text(), re.M).group(1)
        assert guid not in guids, (relative, guids.get(guid))
        guids[guid] = relative
    for meta in PACKAGE.rglob("*.meta"):
        assert Path(str(meta)[: -len(".meta")]).exists(), meta.relative_to(PACKAGE)


def test_package_manifest_names_its_unity_dependencies():
    manifest = json.loads((PACKAGE / "package.json").read_text())
    assert manifest["name"] == "io.github.m-colley.embodied-ui"
    assert {"com.unity.inputsystem", "com.unity.modules.physics"} <= set(
        manifest["dependencies"]
    )
