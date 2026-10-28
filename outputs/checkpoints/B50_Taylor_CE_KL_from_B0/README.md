# B-50 Taylor + CE + KL Teacher

GitHub B-0를 공통 출발 모델과 frozen KL Teacher로 사용한 CUB-200-2011 결과다.
원래 MLP hidden 폭의 누적 50%를 제거했다. 전체 parameter 제거율과는 다르다.

- [최고 모델 다운로드](https://drive.google.com/file/d/1_5_32L6pbWTzUzhtOdUTIf6S4Pem3ssn/view), 183,759,503 bytes.
- SHA-256: `a8a9051383c54425570ab1da915cc1071781072dfe492d1504dd914175534fae`.
- 선택 segment epoch: 18/20, 전체 입력 stream epoch: 58/80. 선택 범위는 해당 segment의 마지막 quarter다.
- Validation OA: 82.0833%, Test OA: 82.3438%.
- `config.json`, `history.json`, `pruning_spec.json`은 생산자가 저장한 export 원문이다. 원본 `github_history.json`을 저장소 관례에 맞춰 `history.json`으로 배치했다.
- `history.json`의 epoch는 전체 입력 stream의1-based 번호다(30%:1~40,50%:41~60,70%:61~80). accuracy는0~1, `metrics.json`의 `_pct` 값은0~100이다. `native_row`는 원래 학습 기록이다.
- 가중치는 Git에서 제외한다. 파일을 내려받을 경우 이 폴더의 `best.pth`에 둔다.
- 이 모델은 기존 L1 B-30과 다른 Taylor 구조 정보 형식을 사용한다. `experiments/cub_taylor_kd/checkpoints.py`의 `load_teacher_export`를 사용한다.

원문 Test CSV와 전체 실험 조건은 [CUB 결과](../../../docs/results/cub_taylor_ce_kl_from_b0/README.md)에서 확인한다.
