# CUB-200-2011 B-30 Teacher 학습 결과: 원본 사전학습 모델 기반

DINOv3 ConvNeXt-Base의 LVD-1689M 사전학습 모델에 pruning을 적용한 뒤, CUB-200-2011로 fine-tuning한 pilot 결과입니다.

각 블록의 MLP 내부 채널을 L1 중요도 기준으로 약 30% 제거했습니다. 전체 parameter의 30%를 제거했다는 의미는 아닙니다.

## 실험 순서

원본 LVD-1689M 사전학습 모델 → MLP 내부 채널 pruning → CUB-200-2011 fine-tuning

CUB 학습을 완료한 B-0에서 pruning한 실험과 구분합니다.

## 체크포인트 다운로드

[B-30 Teacher 최고 체크포인트 다운로드](https://drive.google.com/file/d/1LGcKYEWK0bZgnALESDA9bvLz_rvmslx9/view?usp=drive_link)

[B30_init.pth 다운로드](https://drive.google.com/file/d/1-Kzcmr5xhSN1Gd4DdGQmSMBGbbMs5On-/view?usp=drive_link)

- 최고 체크포인트 파일명: `best.pth`
- 초기 체크포인트 파일명: `B30_init.pth`
- `best.pth`는 pruning 이후 CUB 학습을 완료한 모델입니다.
- 가중치는 Git에 포함하지 않고 Drive로 공유합니다.

## 검증 결과

| 항목 | 값 |
| --- | ---: |
| 학습 seed | 0 |
| 전체 학습 epoch | 20 |
| 선택된 epoch | |
| 최고 검증 정확도 | 45.33% |
| 해당 검증 loss | 2.9838 |
| 복원 후 검증 정확도 | 45.33% |
| 복원 후 검증 loss | 2.9838 |

검증 정확도가 가장 높은 모델을 선택했으며, 동률이면 검증 loss가 낮은 모델을 선택했습니다. 테스트 데이터는 평가에 사용하지 않았습니다.

## 관련 파일

- `config.json`: 학습 설정, 데이터로더 정보와 실행 환경
- `history.json`: epoch별 학습과 검증 기록
- `pruning_spec.json`: 축소된 구조와 유지한 채널 정보
- `notebooks/teacher_prunning.ipynb`: 초기 pruning 코드
- `notebooks/teacher_B30_original_pretrained_completed.ipynb`: fine-tuning 코드

## 복원 방법

DINOv3 ConvNeXt-Base 구조에 200클래스 분류층을 구성합니다.

`pruning_spec.json`에 맞춰 각 블록의 `pwconv1`과 `pwconv2` 내부 채널 수를 변경한 뒤, `best.pth` 안의 `model_state_dict`를 복원합니다.

이미 pruning된 체크포인트이므로, 복원 과정에서 pruning을 다시 적용하지 않습니다.

이 결과는 단일 seed의 pilot 결과이며, 최종 반복 실험 결과는 아닙니다.
