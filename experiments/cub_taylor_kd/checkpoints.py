"""다운로드한 CUB Teacher와 Student 체크포인트를 검증하고 복원한다."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from .bootstrap import load_runtime


def _verified_bytes(path: str | Path, expected_sha256: str) -> bytes:
    if (len(expected_sha256) != 64
            or any(character not in '0123456789abcdef' for character in expected_sha256)):
        raise ValueError('artifacts.json에 기록된 SHA-256을 지정하세요.')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('다운로드한 체크포인트의 SHA-256이 다릅니다.')
    return raw


def load_teacher_export(path, *, expected_sha256, repo_root, extract_parent='/content'):
    """기록된 best.pth의 실제 layout과 tensor digest로 CPU 모델을 복원한다."""
    raw = _verified_bytes(path, expected_sha256)
    load_runtime('teacher', repo_root=repo_root, drive_project_root=repo_root,
                 extract_parent=extract_parent)
    from lightio.runtime_bootstrap import cub_runtime_bindings
    cub_runtime_bindings()
    from phaseb.native import module
    from lightio.github_source import restore_export
    import torch

    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if (value['num_classes'] != 200
            or value['pruning_spec'].get('schema') != 'original_index_layout.v1'
            or value['pruning_spec'].get('target_percent') not in (30, 50, 70)):
        raise ValueError('CUB Taylor Teacher export 형식이 아닙니다.')
    names = value['config']['source_B0']['class_names']
    if value['class_names'] != {index: name for index, name in enumerate(names)} or len(names) != 200:
        raise ValueError('CUB 클래스 번호와 이름의 연결이 다릅니다.')
    models, native_io = module('cub200', 'models'), module('cub200', 'io')
    spec = value['config']['source_B0']['model_spec']

    def factory():
        return models.build_from_spec(spec)

    model = restore_export(value, factory, native_io)
    return model.eval(), value


def load_student_checkpoint(path, *, expected_sha256, repo_root, extract_parent='/content'):
    """Native Student의 model_spec, model_state와 digest를 사용해 복원한다."""
    raw = _verified_bytes(path, expected_sha256)
    load_runtime('kd', repo_root=repo_root, drive_project_root=repo_root,
                 extract_parent=extract_parent)
    from phaseb.native import config, module
    config('cub200')
    import torch

    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    spec = value['model_spec']
    if spec.get('kind') != 'student' or spec.get('num_classes') != 200 or spec.get('test_fixture'):
        raise ValueError('실제 CUB Student 체크포인트가 아닙니다.')
    native_io = module('cub200', 'io')
    if native_io.model_state_identity(value['model_state']) != value['model_digest']:
        raise ValueError('Student tensor digest가 다릅니다.')
    model = module('cub200', 'models').build_from_spec(spec)
    model.load_state_dict(value['model_state'], strict=True)
    return model.eval(), value
