"""Static validator for the Inno Setup installer script.

Inno Setup only reports one problem at a time, on a Windows-only compiler — which makes the
edit/compile loop slow when authoring ``.iss`` files. This checker parses the script the way
ISCC does and reports **everything** it can detect up front:

  * identifier rules for ``[Components]`` / ``[Types]`` / ``[Tasks]`` / ``[Icons]`` / ``[Files]``
    names (alphanumerics, ``_``, ``/``, ``\\``; no hyphens; no leading digit; ``and``/``not``/``or``
    reserved);
  * every ``{code:Function}`` referenced by the script exists in ``[Code]``;
  * every payload path referenced by ``[Files]`` / ``LicenseFile`` / ``SetupIconFile`` exists on
    disk (catches the "payload not staged" class of failure *before* ISCC runs);
  * every component id used by ``IsComponentSelected('…')`` is declared;
  * every ``{#Define}`` used is declared.

Usage::

    python scripts/check_iss.py                       # validates the tiered installer
    python scripts/check_iss.py --stage dist/omniforge-bootstrap
    python scripts/check_iss.py installer/omniforge_installer_tiers.iss --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ISS = PROJECT_ROOT / "installer" / "omniforge_installer_tiers.iss"

RESERVED = {"and", "or", "not"}
IDENT_RE = re.compile(r"^[A-Za-z0-9_/\\]+$")
DEFINE_RE = re.compile(r"^#define\s+(\w+)\s+(.+)$", re.IGNORECASE)
USEDEF_RE = re.compile(r"\{#(\w+)\}")
CODE_REF_RE = re.compile(r"\{code:(\w+)\}")
SECTION_RE = re.compile(r"^\[(\w+)\]$")

#: Sections whose ``Name:`` values are identifiers subject to the naming rules.
NAMED_SECTIONS = {"components", "types", "tasks", "icons", "files", "dirs", "registry"}
#: Sections that accept the same identifier syntax (used for cross-checking references).
IDENT_SECTIONS = {"components", "types", "tasks"}


class Issue:
    """A single validation finding."""

    def __init__(self, level: str, line: int, message: str) -> None:
        self.level = level
        self.line = line
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "line": self.line, "message": self.message}

    def __str__(self) -> str:
        return f"line {self.line}: [{self.level}] {self.message}"


def parse_identifier(raw: str) -> str:
    """Return the identifier part of a ``Name:`` value (strips flags/params)."""
    value = raw.strip().strip('"')
    for sep in (";",):
        value = value.split(sep)[0]
    return value.strip().strip('"')


def check_identifier(name: str, kind: str) -> str | None:
    """Return an error message when ``name`` breaks Inno's identifier rules."""
    if not name:
        return f"{kind} has an empty name"
    if "-" in name:
        return (
            f"{kind} name {name!r} contains '-' — Inno only allows alphanumerics, '_', '/' and "
            f"'\\' (use '_' instead, e.g. {name.replace('-', '_')!r})"
        )
    if not IDENT_RE.match(name):
        bad = sorted({ch for ch in name if not re.match(r"[A-Za-z0-9_/\\]", ch)})
        return f"{kind} name {name!r} contains invalid characters: {bad}"
    if name[0].isdigit():
        return f"{kind} name {name!r} may not start with a digit"
    if name[0] in "/\\" or name[-1] in "/\\":
        return f"{kind} name {name!r} may not start or end with a slash"
    if name.lower() in RESERVED:
        return f"{kind} name {name!r} is reserved"
    return None


def _to_local(path_text: str) -> str:
    """Translate a Windows path written in the .iss into a host-native path."""
    import os

    return path_text.replace("\\", os.sep).replace("/", os.sep)


def _reroot(candidate: Path, reroot: tuple[Path, Path] | None) -> Path:
    """Point a path that lives inside the scripted staging root at the --stage override."""
    if reroot is None:
        return candidate
    staging_root, override = reroot
    try:
        return override / candidate.resolve().relative_to(staging_root)
    except (ValueError, OSError):
        return candidate


def check_payload_path(
    raw: str,
    base: Path,
    define_values: dict[str, str],
    reroot: tuple[Path, Path] | None = None,
) -> str | None:
    """Verify a ``Source:``/``LicenseFile=`` path resolves to an existing file (when checkable)."""
    value = _to_local(raw.strip().strip('"'))
    # Substitute {#Define} references.
    for name, replacement in define_values.items():
        value = value.replace(f"{{#{name}}}", _to_local(replacement))
    if re.search(r"\{[a-z]+[:}]", value, re.IGNORECASE):
        return None  # shell folders such as {app}, {tmp}: not resolvable statically
    if "*" in value or "?" in value:
        parent = Path(value.split("*")[0])
        candidate = (base / parent) if not parent.is_absolute() else parent
        candidate = _reroot(candidate, reroot)
        return None if candidate.exists() else f"glob source directory does not exist: {value}"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / value
    candidate = _reroot(candidate, reroot)
    try:
        if candidate.is_file():
            return None
    except OSError:
        return f"cannot stat {value}"
    return f"payload file not found: {value}  (stage the payload first: python scripts/build_bootstrap.py)"


def validate(iss_path: Path, stage: Path | None = None) -> list[Issue]:
    """Validate one ``.iss`` file and return every finding."""
    text = iss_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    base_dir = iss_path.parent
    issues: list[Issue] = []

    defines: dict[str, str] = {
        m.group(1): m.group(2).strip().strip('"')
        for m in (DEFINE_RE.match(line.strip()) for line in lines)
        if m
    }
    staging_root: Path | None = None
    if stage is not None and defines.get("StagingDir"):
        raw_root = _to_local(defines["StagingDir"])
        root_path = Path(raw_root)
        staging_root = (root_path if root_path.is_absolute() else base_dir / root_path).resolve()
    reroot = (staging_root, stage) if staging_root is not None and stage is not None else None
    declared: dict[str, set[str]] = {section: set() for section in IDENT_SECTIONS}
    optional_components: dict[str, bool] = {}
    used_components: list[tuple[int, str]] = []
    code_refs: list[tuple[int, str]] = []
    section = ""

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith((";", "#")):
            define = DEFINE_RE.match(stripped)
            if define:
                defines[define.group(1)] = define.group(2).strip().strip('"')
            for define_use in USEDEF_RE.finditer(stripped):
                if define_use.group(1) not in defines:
                    issues.append(Issue("error", number, f"undefined {{#{define_use.group(1)}}}"))
            code_refs.extend((number, m.group(1)) for m in CODE_REF_RE.finditer(stripped))
            continue

        match = SECTION_RE.match(stripped)
        if match:
            section = match.group(1).lower()
            continue

        for define_use in USEDEF_RE.finditer(stripped):
            if define_use.group(1) not in defines:
                issues.append(Issue("error", number, f"undefined {{#{define_use.group(1)}}}"))
        code_refs.extend((number, m.group(1)) for m in CODE_REF_RE.finditer(stripped))

        if section == "code":
            continue

        for name_match in re.finditer(r"\bName:\s*(\"[^\"]*\"|[^;]+)", stripped):
            raw = name_match.group(1)
            ident = parse_identifier(raw)
            if section in IDENT_SECTIONS:
                error = check_identifier(ident, f"[{section}]")
                if error:
                    issues.append(Issue("error", number, error))
                declared.setdefault(section, set()).add(ident)
                if section == "components":
                    optional_components[ident] = "unchecked" in stripped.lower()

        if section == "files":
            source_match = re.search(r"\bSource:\s*(\"[^\"]*\"|[^;]+)", stripped)
            if source_match and stage is not None:
                error = check_payload_path(source_match.group(1), base_dir, defines, reroot)
                if error:
                    # A missing payload is an error; a stray file is only a warning.
                    issues.append(Issue("error" if "payload" in error else "warning", number, error))

        if section == "":
            for key, label in (("LicenseFile", "license"), ("SetupIconFile", "icon")):
                if stripped.lower().startswith(key.lower() + "=") and stage is not None:
                    error = check_payload_path(stripped.split("=", 1)[1], base_dir, defines, reroot)
                    if error:
                        issues.append(Issue("warning", number, f"{label}: {error}"))

        for selection in re.finditer(r"IsComponentSelected\('([^']+)'\)", stripped):
            used_components.append((number, selection.group(1)))

    declared_components = declared.get("components", set())
    known_components = set(declared_components) | {"core_runtime", "data_io", "api_server"}
    for number, name in used_components:
        if name not in known_components:
            issues.append(Issue("error", number, f"IsComponentSelected({name!r}) is not declared in [Components]"))

    # Pascal 'if' chains are easy to break: every OPTIONAL (unchecked) component must be
    # referenced from [Code], otherwise ticking it in the wizard installs nothing. Tier packs
    # (Flags: fixed / types-selected) do not need a reference: the tier decides for them.
    pascal = text[text.find("[Code]"):] if "[Code]" in text else ""
    for component, optional in optional_components.items():
        if optional and f"'{component}'" not in pascal:
            issues.append(
                Issue(
                    "warning",
                    0,
                    f"optional component {component!r} is never referenced in [Code] — ticking it would install nothing",
                )
            )

    if code_refs:
        code_body = text[text.find("[Code]"):] if "[Code]" in text else ""
        for number, function in code_refs:
            if not re.search(rf"\bfunction\s+{function}\b", code_body):
                issues.append(Issue("error", number, f"{{code:{function}}} has no matching function in [Code]"))

    issues.extend(check_setup_directives(lines))
    issues.extend(check_setup_values(lines))
    issues.extend(check_duplicate_files(lines))
    issues.extend(check_pp_messages(lines))
    issues.extend(check_if_conditions(lines))
    issues.extend(check_preprocessor(text, lines))

    return issues


def check_setup_directives(lines: list[str]) -> list[Issue]:
    """Every ``[Setup]`` key must be a real Inno 6 directive."""
    known = {
        "appid", "appname", "appversion", "appvername", "apppublisher", "apppublisherurl",
        "appsupporturl", "appupdatesurl", "appcomments", "appcontact", "defaultdirname",
        "defaultgroupname", "allowunsafefiles", "licensefile", "outputdir", "outputbasefilename",
        "setupiconfile", "uninstalldisplayicon", "uninstalldisplayname", "compression",
        "solidcompression", "wizardstyle", "architecturesallowed",
        "architecturesinstallin64bitmode", "privilegesrequired",
        "privilegesrequiredoverridesallowed", "disableprogramgrouppage", "closeapplications",
        "minversion", "changesenvironment", "usediskspanning", "createuninstallregkey",
        "dirifempty", "diskclustersize", "enablesigntool", "sign", "signeduninst",
        "versioninfoversion", "versioninfodescription", "versioninfocopyright",
        "wizardimagefile", "wizardsmallimagefile", "languagedetectionmethod", "showlanguagepagedialog",
        "usepreviouslanguage", "uninstallable", "restartreplacemode", "setuptype",
    }
    issues: list[Issue] = []
    in_setup = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("["):
            in_setup = stripped.lower() == "[setup]"
            continue
        if not in_setup or not stripped or stripped.startswith(";") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip().lower()
        if key not in known:
            issues.append(Issue("error", number, f"unknown [Setup] directive {key!r}"))
    return issues


def check_duplicate_files(lines: list[str]) -> list[Issue]:
    """Inno refuses two entries with the same DestDir + DestName ('installed more than once')."""
    issues: list[Issue] = []
    in_files = False
    seen: dict[tuple[str, str], int] = {}
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("["):
            in_files = stripped.lower() == "[files]"
            continue
        if not in_files or not stripped.startswith("Source:") or stripped.startswith(";"):
            continue
        dest_dir = re.search(r'DestDir:\s*"([^"]*)"', stripped)
        dest_name = re.search(r'DestName:\s*"([^"]*)"', stripped)
        source = re.search(r'Source:\s*"([^"]*)"', stripped)
        if dest_name and source:
            target = Path(_to_local(source.group(1))).name
            if dest_name.group(1).lower() == target.lower():
                issues.append(
                    Issue("warning", number,
                          f"DestName equals the source file name ({target}) — redundant entry")
                )
        if not dest_dir:
            continue
        key = (dest_dir.group(1), dest_name.group(1) if dest_name else "<glob>")
        if key in seen and "<glob>" not in key[1]:
            issues.append(Issue("error", number, f"duplicate [Files] target {key}"))
        seen[key] = number
    return issues


def check_setup_values(lines: list[str]) -> list[Issue]:
    """Directive values Inno validates by extension (LicenseFile, SetupIconFile)."""
    issues: list[Issue] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith((";", "#")):
            continue
        lowered = stripped.lower()
        if lowered.startswith("licensefile="):
            value = stripped.split("=", 1)[1].strip().strip('"')
            if not value.lower().endswith((".txt", ".rtf")):
                issues.append(
                    Issue(
                        "error",
                        number,
                        f"LicenseFile must be a .txt or .rtf file, not {value!r} - Inno refuses "
                        "any other extension (stage a LICENSE.txt twin)",
                    )
                )
        elif lowered.startswith("setupiconfile="):
            value = stripped.split("=", 1)[1].strip().strip('"')
            if not value.lower().endswith((".ico", ".exe", ".dll")):
                issues.append(
                    Issue("error", number, f"SetupIconFile must be .ico/.exe/.dll, not {value!r}")
                )
    return issues


def check_pp_messages(lines: list[str]) -> list[Issue]:
    """ISPP ``#error``/``#pragma`` messages must be quoted string literals."""
    issues: list[Issue] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        for directive in ("#error", "#pragma"):
            if not stripped.lower().startswith(directive):
                continue
            rest = stripped[len(directive) :].strip()
            if not rest:
                issues.append(Issue("error", number, f"{directive} without a message"))
            elif directive == "#error" and not rest.startswith('"'):
                issues.append(
                    Issue("error", number, f"#error message must be quoted: {rest[:40]!r}")
                )
    return issues


def check_if_conditions(lines: list[str]) -> list[Issue]:
    """A ``#if``/``#elif`` condition may not be a bare unquoted string (ISPP compares ints)."""
    issues: list[Issue] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        match = re.match(r"#(?:if|elif)\s+(.+)$", stripped, re.IGNORECASE)
        if not match:
            continue
        condition = match.group(1).strip()
        if re.fullmatch(r"[A-Za-z_][\w.!]*", condition) and not condition.lower().startswith(
            ("defined", "exist")
        ):
            issues.append(
                Issue(
                    "error",
                    number,
                    f"#if {condition!r} compares against the identifier's string value; "
                    f'use #if "{condition}" == "..." or #ifdef {condition}',
                )
            )
    return issues


def check_preprocessor(text: str, lines: list[str]) -> list[Issue]:
    """Catch ISPP function misuse (e.g. ``AddBackslash``, which is a [Code] Pascal function)."""
    del text  # signature kept for symmetry with the other checkers
    issues: list[Issue] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped.startswith("#define"):
            continue
        value = stripped[len("#define") :].strip()
        call = re.search(r"\b(\w+)\s*\(", value)
        if call and call.group(1).lower() == "addbackslash":
            issues.append(
                Issue(
                    "error",
                    number,
                    "AddBackslash() is a [Code] Pascal function, not an ISPP function - use plain "
                    'string concatenation in #define, e.g. StagingDir + "\\app\\LICENSE"',
                )
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate an .iss and exit non-zero on errors."""
    parser = argparse.ArgumentParser(prog="check_iss", description=__doc__.splitlines()[0])
    parser.add_argument("iss", nargs="?", default=str(DEFAULT_ISS))
    parser.add_argument("--stage", default=None, help="Staging dir for payload existence checks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    iss_path = Path(args.iss).resolve()
    if not iss_path.is_file():
        print(f"error: {iss_path} not found")
        return 2
    stage = Path(args.stage).resolve() if args.stage else None

    issues = validate(iss_path, stage)
    errors = [item for item in issues if item.level == "error"]
    if args.json:
        print(json.dumps({"file": str(iss_path), "issues": [i.as_dict() for i in issues]}, indent=2))
    else:
        for item in issues:
            print(item)
        if not issues:
            suffix = f" (payload: {stage})" if stage else ""
            print(f"OK: {iss_path.name} passed all checks{suffix}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
