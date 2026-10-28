"""기록된 CUB 모델 목록을 보여 주고 요청한 파일만 Google Drive에서 받는다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def destination(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('저장소 밖의 체크포인트 경로는 허용하지 않습니다.')
    return path


def download(service, record, root):
    from googleapiclient.http import MediaIoBaseDownload

    target = destination(root, record['destination'])
    if target.exists():
        if target.is_file() and target.stat().st_size == record['bytes'] and file_sha256(target) == record['sha256']:
            print('VERIFIED_EXISTING', record['key'], target)
            return
        raise FileExistsError('기존 파일을 보존합니다. 내용을 확인하세요: ' + str(target))
    fields = 'id,size,sha256Checksum,version,trashed'
    before = service.files().get(fileId=record['file_id'], fields=fields).execute()
    if before.get('trashed') or int(before['size']) != record['bytes']:
        raise ValueError('Drive 원본의 상태 또는 크기가 완료 기록과 다릅니다.')
    if before.get('sha256Checksum') and before['sha256Checksum'] != record['sha256']:
        raise ValueError('Drive 원본의 checksum이 완료 기록과 다릅니다.')
    expected_version = record.get('provider_version_at_completion')
    if expected_version and before.get('version') != expected_version:
        raise ValueError('Drive 원본의 version이 완료 기록과 다릅니다.')
    scratch_root = root / 'codex_ws' / 'runs'
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='cub_checkpoint_', dir=scratch_root)) / target.name
    with temporary.open('wb') as handle:
        transfer = MediaIoBaseDownload(handle, service.files().get_media(fileId=record['file_id']),
                                      chunksize=8 * 1024 * 1024)
        finished = False
        while not finished:
            _, finished = transfer.next_chunk()
        handle.flush()
        os.fsync(handle.fileno())
    after = service.files().get(fileId=record['file_id'], fields=fields).execute()
    if after != before or temporary.stat().st_size != record['bytes'] or file_sha256(temporary) != record['sha256']:
        raise ValueError('다운로드 중 원본 변경 또는 checksum 불일치를 확인했습니다. 임시 파일을 보존합니다.')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError('다운로드 중 목적 파일이 생겼습니다. 덮어쓰지 않습니다.')
    os.link(temporary, target)
    temporary.unlink()
    print('DOWNLOADED_AND_VERIFIED', record['key'], target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--kind', choices=('all', 'teacher', 'student'), default='all')
    parser.add_argument('--key', help='artifacts.json의 정확한 key 하나를 선택한다.')
    parser.add_argument('--download', action='store_true', help='지정해야 실제 다운로드한다.')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = root / 'docs/results/cub_taylor_ce_kl_from_b0/artifacts.json'
    models = json.loads(manifest.read_text(encoding='utf-8'))['models']
    selected = [row for row in models if (args.kind == 'all' or row['kind'] == args.kind)
                and (args.key is None or row['key'] == args.key)]
    if not selected:
        raise ValueError('선택한 모델이 없습니다.')
    for row in selected:
        destination(root, row['destination'])
        print(row['key'], row['bytes'], row['sha256'], row['url'])
    if not args.download:
        print('LIST_ONLY: 실제 다운로드는 --download 옵션으로 요청한다.')
        return
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
    service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    for row in selected:
        download(service, row, root)


if __name__ == '__main__':
    main()
