# CUB B-0 기반 Taylor + CE + KL 실행 자료

팀 저장소의 CUB B-0를 출발 모델과 frozen KL Teacher로 사용한다. 실행한 과학 코드와 설정을 동결 ZIP으로 보존하고, 저장소에서 사용할 경로 설정과 체크포인트 복원 함수를 제공한다.

## 노트북

| 파일 | 역할 |
|---|---|
| [teacher_B30_50_70_Taylor_CE_KL_from_B0_colab.ipynb](../../notebooks/teacher_B30_50_70_Taylor_CE_KL_from_B0_colab.ipynb) | B-0에서 누적 30%, 50%, 70%까지 Taylor 가지치기와 CE+KL 회복 학습을 수행한다. |
| [kd_Taylor_CE_KL_from_B0_colab.ipynb](../../notebooks/kd_Taylor_CE_KL_from_B0_colab.ipynb) | B-0/B-30/B-50/B-70 Teacher로 MobileNetV3-Large를 각각 세 seed 학습한다. |

두 노트북은 각각 전용 Colab Python 커널에서 사용한다. 같은 커널에 Teacher와 KD의 `lightio` 모듈을 섞으면 중단한다. 공유 사본은 출력과 실행 번호를 비웠고 `START_TRAINING=False`, `AUTO_RELEASE=False`다. 기본 상태에서 학습을 시작하지 않는다.

## 저장소에서 실행하는 순서

1. 이 자료를 올린 브랜치를 Colab의 `/content/ondeviceAI`에 clone한다. 브랜치명은 실제 게시한 이름을 사용한다. 노트북만 GitHub에서 여는 동작은 저장소 파일을 clone하지 않는다.
2. 옵션 셀의 `REPO_ROOT`와 `DRIVE_PROJECT_ROOT`를 확인한다. 기본 Drive root는 `/content/drive/MyDrive/온디바이스AI`다. 필요한 원본 CUB 데이터와 기존 모델을 읽을 수 있는 Google 계정을 사용한다.
3. setup 셀을 실행한다. 저장소의 `runtime/` ZIP을 먼저 읽고 SHA-256, CRC, 소스 manifest와 code ID를 검사한다. 저장소 ZIP이 없을 때는 기존 Drive의 정확한 ZIP 경로를 확인한다.
4. 실행할 작업의 입력과 출력 위치를 확인한 뒤 `START_TRAINING=True`로 실행한다. 이미 완료한 결과를 읽는 목적이면 학습 셀을 다시 실행하지 않는다. 동일 실행을 재개하는 경우에만 `RESUME=True`를 사용한다.
5. KD는 완료된 Teacher sequence와 Test, 최근 runtime 종료 관측을 확인한다. `TERMINAL_OBSERVATION`에 과거 날짜를 미리 채우지 않는다. 필수 필드는 KD 노트북의 설명을 따른다.

실제 실행 함수의 인수와 본문은 원본 노트북과 같다. 정리 과정에서 바꾼 범위는 설명, 경로 연결, 출력 제거와 공유 기본값이다. 학습, source checkpoint, Hard 선정, Test 잠금, 소유권과 재개 검사는 runtime 내부 원본을 사용한다.

## 외부 입력과 접근 범위

- CUB 원본 이미지는 Git에 포함하지 않는다. `DATA_ROOT=None`이면 원래 runtime의 준비 경로를 사용한다. 다른 위치를 지정할 때도 공유 split와 메타데이터 검증을 통과해야 한다.
- B-0는 [기존 체크포인트](https://drive.google.com/file/d/1xwR2mviCdrSYJ-n2eZuFbd14reVcXuiq/view)이며, 이 실험에서 고정한 SHA-256은 `3ceffe27b7d41a711937961e117a28d95b2d5611be4dc473969155b78620f207`이다. B-0를 다시 학습하지 않는다.
- KD는 기존 Native Student의 seed별 초기화와 Student-only baseline을 검증 후 재사용한다. 저장소에는 큰 원본 초기화와 baseline 가중치를 복사하지 않는다.
- 완료 실행의 모델과 결과는 [산출물 목록](../../docs/results/cub_taylor_ce_kl_from_b0/artifacts.json), 전체 raw 근거는 [결과 문서](../../docs/results/cub_taylor_ce_kl_from_b0/README.md)에 연결한다.
- Drive 링크는 접근 권한이 있는 계정에서 사용한다. 이번 정리는 파일의 공유 권한을 바꾸지 않았다. 현재 새 selected 모델 15개는 공개 공유되지 않은 상태다.

Google Colab의 PyTorch, torchvision, NumPy, Pillow, tqdm, google-auth, google-api-python-client를 사용한다. 실행 중인 Colab의 torch 환경을 임의로 업그레이드하지 않고 runtime 검사 결과를 확인한다. 루트 `requirements.txt`는 원래 로컬 실습 환경 기록이며 완료 Colab 실행 환경과 구분한다.

## 동결 실행 코드

| Runtime ZIP | bytes | code ID |
|---|---:|---|
| `GIT_B0_TAYLOR_30_50_70_v1_2.zip` | 2,367,996 | `81aefa8e76aa6e245c427745c532d0545ebd072d14669da21e0a057cdcde5a7d` |
| `KD_OFFICIAL_FOLLOWUP_v1.zip` | 2,417,035 | `872e1a3560df5dce08a03efdf0932763299545a61921f95418a3b2a6ef9545af` |

ZIP SHA와 원본 위치는 [runtime_manifest.json](runtime_manifest.json)에 기록한다. ZIP에는 실행에 필요한 Python 소스, 구조와 split 메타데이터 및 원래 라이선스가 들어 있다. 원본 이미지와 학습된 가중치는 없다. 두 ZIP의 공통 코드를 합치지 않아 원래 code identity를 유지한다.

코드 검토는 ZIP을 별도 작업 폴더에 풀어 진행할 수 있다. `lightio/github_source.py`는 B-0 검증과 Teacher export 복원, `lightio/taylor.py`는 Taylor 중요도, `lightio/recovery_loss.py`는 CE+KL, `lightio/extend_runner.py`는 Teacher 단계 진행, KD ZIP의 `lightio/kd_runner.py`는 Student 학습 진입점이다. 동결 kernel은 `vendor_base/followup_iterative_v2/training.py`에 있다.

ZIP 내부의 `README.md`, `prepared_only`, 패키지 acceptance는 준비 당시 기록이다. 완료 상태는 새 결과 폴더의 실제 `COMPLETE`와 CSV 검증으로 판단한다. ZIP 안의 `build_package.py`는 이 자료의 실행 경로에서 호출하지 않는다.

## 체크포인트 받기와 복원

기존 저장소처럼 Git에는 JSON과 다운로드 링크를 넣고 `.pt`, `.pth`는 제외한다. Teacher는 각 `outputs/checkpoints/B{비율}_Taylor_CE_KL_from_B0/best.pth`, Student는 `outputs/checkpoints/KD_Taylor_CE_KL_from_B0/teacher_B{비율}/seed_{seed}/best.pt`에 둘 수 있다.

모델 목록만 확인하는 명령은 다음과 같다. 네트워크나 인증을 호출하지 않는다.

```bash
python tools/download_cub_checkpoints.py
python tools/download_cub_checkpoints.py --kind teacher
```

실제로 받을 때는 Google 인증이 준비된 Colab 또는 환경에서 `--download`를 지정한다. `--key teacher_30` 또는 `--key KD_B0_seed42`로 파일 하나를 선택할 수 있다. 도구는 원본 ID와 크기, 가능한 provider checksum과 version을 검사하고, 전체 다운로드 SHA-256이 일치한 파일만 최종 위치에 둔다. 기존 다른 파일을 덮어쓰지 않는다.

```bash
python tools/download_cub_checkpoints.py --key teacher_30 --download
```

Teacher `best.pth`에는 `model_state_dict`와 `pruning_spec`이 있다. Taylor의 `pruning_spec.blocks`는 original-index layout list이며, 과거 L1 노트북의 dictionary 형식과 다르다. 기존 L1 복원 함수를 그대로 사용하지 않고 아래 wrapper를 쓴다. wrapper는 실제 채널 폭과 model digest를 검증한다.

```python
from experiments.cub_taylor_kd.checkpoints import load_teacher_export

model, metadata = load_teacher_export(
    "/content/ondeviceAI/outputs/checkpoints/B30_Taylor_CE_KL_from_B0/best.pth",
    expected_sha256="80a3476dc3c5147661b0c3b3db3e5688bb6fcc1d12ea5c2b8cebb4228a249f86",
    repo_root="/content/ondeviceAI",
)
```

Student `best.pt`는 `model_state`, `model_spec`, `model_digest`, `contract`를 사용하는 Native 형식이다. 다른 전용 커널에서 `load_student_checkpoint`를 같은 인수 방식으로 사용한다. 가중치 파일을 PC로 내려받아 재검증하지 않았으므로 이번 검증을 모델 tensor 복원 또는 추론 검증으로 해석하지 않는다.

## Git 포함 범위

두 노트북, helper, exact runtime ZIP 두 개, Teacher와 Student의 config/history/metrics/artifacts, Test CSV와 완료 근거를 포함한다. 원본 이미지, 모델 바이너리, optimizer resume, 캐시와 `codex_ws/`는 제외한다. `.gitignore` 예외는 필요한 metadata 폴더와 작은 source ZIP 두 개로 한정한다.
