"""저장소 또는 Drive의 동결 ZIP을 검증하고 전용 커널에 연결한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile


_LOADED: tuple[str, Path] | None = None
_MODULE_ROOTS = ("lightio", "followup_iterative_v2", "phaseb", "b1kd", "nskd")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source(root: Path, expected_code_id: str) -> None:
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf8"))
    code_id = hashlib.sha256(
        json.dumps(manifest["sha256"], sort_keys=True, ensure_ascii=False,
                   allow_nan=False).encode("utf8")
    ).hexdigest()
    if code_id != expected_code_id or manifest["code_id"] != expected_code_id:
        raise ValueError("동결 실행 패키지의 code ID가 다릅니다.")
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != expected:
            raise ValueError("동결 실행 소스가 다릅니다: " + name)


def load_runtime(
    runtime: str,
    *,
    repo_root: str | Path,
    drive_project_root: str | Path,
    extract_parent: str | Path = "/content",
) -> Path:
    """정확한 ZIP과 code ID를 확인하고 추출된 패키지 루트를 반환한다.

    실행 함수나 데이터, 모델을 읽지 않는다. Teacher와 KD는 별도 Python
    커널에서 연결하며 같은 패키지의 setup 셀 재실행만 허용한다.
    """
    global _LOADED
    repo_root = Path(repo_root).expanduser().resolve()
    package_root = Path(__file__).resolve().parent
    expected_root = repo_root / "experiments" / "cub_taylor_kd"
    if package_root != expected_root.resolve():
        raise ValueError("bootstrap 모듈과 REPO_ROOT가 다른 저장소를 가리킵니다.")
    registry = json.loads((package_root / "runtime_manifest.json").read_text(encoding="utf8"))
    if registry.get("schema_version") != 1 or runtime not in registry["runtimes"]:
        raise ValueError("지원하는 runtime은 teacher 또는 kd입니다.")
    record = registry["runtimes"][runtime]
    repository_zip = package_root / "runtime" / record["file"]
    drive_zip = Path(drive_project_root).expanduser() / record["drive_relative_path"]
    package = repository_zip if repository_zip.is_file() else drive_zip
    if not package.is_file():
        raise FileNotFoundError(
            "동결 ZIP이 없습니다. 공유 브랜치를 clone하고 REPO_ROOT를 확인하세요. "
            f"확인한 경로: {repository_zip}, {drive_zip}"
        )
    if package.stat().st_size != record["bytes"] or _sha(package) != record["sha256"]:
        raise ValueError("동결 ZIP의 크기 또는 SHA-256이 다릅니다: " + str(package))
    active = [name for name in sys.modules
              if any(name == prefix or name.startswith(prefix + ".")
                     for prefix in _MODULE_ROOTS)]
    if _LOADED is not None:
        loaded_runtime, root = _LOADED
        if loaded_runtime != runtime:
            raise RuntimeError("Teacher와 KD는 서로 다른 전용 Python 커널에서 실행하세요.")
        for name in active:
            origin = getattr(sys.modules[name], "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError("다른 실행 패키지의 모듈이 섞였습니다: " + name)
        _verify_source(root, record["code_id"])
        return root
    if active:
        raise RuntimeError(
            "기존 연구 모듈이 없는 전용 Python 커널이 필요합니다: " + ", ".join(active[:5])
        )
    extract_parent = Path(extract_parent).expanduser().resolve()
    if not extract_parent.is_dir():
        raise FileNotFoundError("추출 상위 폴더가 없습니다: " + str(extract_parent))
    root = Path(tempfile.mkdtemp(prefix=f"cub_{runtime}_frozen_", dir=extract_parent)).resolve()
    with zipfile.ZipFile(package) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("동결 ZIP의 CRC 또는 중복 경로 검증에 실패했습니다.")
        if sum(member.file_size for member in members) != record["uncompressed_bytes"]:
            raise ValueError("동결 ZIP의 압축 해제 크기가 다릅니다.")
        for member in members:
            name = PurePosixPath(member.filename)
            target = (root / member.filename).resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                    or ":" in member.filename or stat.S_ISLNK(member.external_attr >> 16)
                    or not target.is_relative_to(root)):
                raise ValueError("허용하지 않는 ZIP 경로입니다: " + member.filename)
        archive.extractall(root)
    _verify_source(root, record["code_id"])
    sys.path[:0] = [str(root), str(root / "vendor_base")]
    _LOADED = (runtime, root)
    print("FROZEN_RUNTIME_VERIFIED", runtime, record["code_id"], str(package))
    return root
