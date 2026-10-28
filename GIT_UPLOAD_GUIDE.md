# CUB 실험 Git 업로드 안내

기준 브랜치는 `dev`, 원래 기준 커밋은 `096e1f12185dca70c51296596499706b3378e1e7`이다. CUB 노트북과 결과 자료를 별도 원격 브랜치 없이 `dev`에 직접 반영한다.

## 포함할 파일

- `.gitignore`, `README.md`, 이 안내문.
- `notebooks/teacher_B30_50_70_Taylor_CE_KL_from_B0_colab.ipynb`.
- `notebooks/kd_Taylor_CE_KL_from_B0_colab.ipynb`.
- `experiments/cub_taylor_kd/`: bootstrap, 복원 helper, 동결 실행 코드 ZIP 두 개와 manifest.
- `docs/results/cub_taylor_ce_kl_from_b0/`: 완료 근거, 결과 요약, 예측 CSV, 검증과 다운로드 목록.
- `outputs/checkpoints/B30_Taylor_CE_KL_from_B0/`, `B50_Taylor_CE_KL_from_B0/`, `B70_Taylor_CE_KL_from_B0/`: Teacher 설정, history, 실제 pruning spec, metrics와 모델 링크.
- `outputs/checkpoints/KD_Taylor_CE_KL_from_B0/`: Student 12개 작업의 설정, history, metrics와 모델 링크.
- `tools/download_cub_checkpoints.py`: 기본 목록 조회, 요청 시 checksum 검증 다운로드.

기존 저장소는 `.pt`, `.pth` 가중치를 Git에서 제외하고 결과 JSON과 Drive 링크를 공유한다. 이 방식을 유지했다. 새 selected 모델은 Teacher 3개와 Student 12개다. 원본 B-0는 기존 체크포인트를 참조한다. 최종 모델은 선택된 best이며, 다음 목표를 잇는 LAST와 optimizer resume 링크는 별도 메타데이터에 있다.

## 로컬 저장소에서 올리기

현재 준비 폴더를 그대로 사용할 경우 다음 순서로 검토하고 실행한다.

```powershell
git status --short
git add -- .gitignore README.md GIT_UPLOAD_GUIDE.md notebooks/teacher_B30_50_70_Taylor_CE_KL_from_B0_colab.ipynb notebooks/kd_Taylor_CE_KL_from_B0_colab.ipynb experiments/cub_taylor_kd docs/results/cub_taylor_ce_kl_from_b0 outputs/checkpoints/B30_Taylor_CE_KL_from_B0 outputs/checkpoints/B50_Taylor_CE_KL_from_B0 outputs/checkpoints/B70_Taylor_CE_KL_from_B0 outputs/checkpoints/KD_Taylor_CE_KL_from_B0 tools/download_cub_checkpoints.py
git diff --cached --stat
git commit -m "feat(cub): B0 기반 Taylor 가지치기와 KD 결과 추가"
git push origin dev
```

이 명령은 현재 로컬 `dev`의 커밋을 GitHub의 `dev`에 반영한다. 사용자 저장소의 Git 인증으로 실행하며 강제 push를 사용하지 않는다. 원격 `dev`가 달라졌으면 먼저 최신 변경을 확인하고 통합한다. 다른 작업 폴더에 변경분 ZIP을 적용할 때는 기준 커밋과 기존 변경을 확인하고 README와 `.gitignore` 변경을 검토한다. 기존 사용자 변경을 덮어쓰지 않는다.

## 검증 범위

노트북 형식과 AST, 비어 있는 실행 출력, 정확한 runtime ZIP SHA와 코드 manifest, 별도 process의 Native 초기화와 계획 조회를 확인했다. Teacher 3개와 Student 12개 history의 선택 규칙, Test CSV 19개와 각 5,794개 ordered ID, 정확도와 CE, Hard 지표를 다시 계산했다. Student 5조건의 세 seed 평균과 표본 SD도 대조했다.

가중치는 이번 준비 작업에서 다운로드하지 않았다. 모델 SHA는 완료 영수증의 값을 보존하고, Drive에서 selected 15개 파일의 ID와 크기, 계정 접근 가능 상태를 확인했다. 새 모델 tensor 복원과 GPU 추론 검증은 하지 않았다. 저장 결과 검증과 새 노트북의 GPU 재실행을 구분한다.

현재 새 selected 모델의 Drive 파일은 공개 공유 상태가 아니다. 다운로드에는 해당 파일에 접근 가능한 Google 계정이 필요하다. 공유 권한을 바꾸지 않았다. 문서 링크와 모델 접근 범위는 별개다.

원본 학습 노트북과 완료 결과는 그대로 보존했다. 새 노트북의 기본값은 학습을 시작하지 않으며, 단순히 결과를 보기 위해 학습 셀을 다시 실행할 필요가 없다. [완료 결과](docs/results/cub_taylor_ce_kl_from_b0/README.md)와 [실행 안내](experiments/cub_taylor_kd/README.md)를 참고한다.
