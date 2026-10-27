# CUB-200-2011 B-30 Teacher 회복 학습 결과: B-0 기반

CUB-200-2011로 fine-tuning을 완료한 B-0 Teacher에 pruning을 적용하고, 같은 학습 데이터로 회복 학습한 pilot 실험입니다.

각 블록의 MLP 내부 채널을 L1 중요도 기준으로 약 30% 제거했습니다. 전체 parameter의 30%를 제거했다는 의미는 아닙니다.

## 실험 순서

LVD-1689M 사전학습 모델 → CUB-200-2011 fine-tuning으로 B-0 생성 → MLP 내부 채널 pruning → CUB-200-2011 회복 학습

원본 사전학습 모델에서 바로 pruning한 실험보다 CUB 학습 과정이 추가되므로, 두 조건을 구분해서 비교해야 합니다.

## 시작 모델

- 모델: DINOv3 ConvNeXt-Base, 200클래스 분류층
- 시작 체크포인트: B-0의 `best.pth`
- B-0 선택 epoch: 10
- B-0 검증 정확도: 83.33%
- B-0 검증 loss: 0.7331
- Pruning 전 재평가에서 같은 정확도와 loss를 확인했습니다.
- Pruning 과정에서 B-0의 학습된 분류층을 유지했습니다.

## 체크포인트 다운로드

[B-30 회복 학습 최고 체크포인트 다운로드](https://drive.google.com/file/d/1gf3sW7J5GyWU47uP-HZH3HwOhERcugz4/view?usp=sharing)

[B30_from_B0_init.pth 다운로드](https://drive.google.com/file/d/1CpRCgyUdwI31C92VzcDmQ6-XiwLsnodQ/view?usp=sharing)

- 최고 체크포인트 파일명: `best.pth`
- 회복 학습 초기 파일명: `B30_from_B0_init.pth`
- 초기 파일은 CUB 학습을 마친 B-0에서 pruning한 직후의 상태입니다.
- 가중치는 Git에 포함하지 않고 Drive로 공유합니다.

## 검증 결과

| 항목 | 값 |
| --- | ---: |
| 학습 seed | 0 |
| 회복 학습 epoch 수 | |
| Pruning 직후 검증 정확도 | |
| Pruning 직후 검증 loss | |
| 회복 학습 선택 epoch | |
| 최고 검증 정확도 | |
| 해당 검증 loss | |
| 복원 후 검증 정확도 | |
| 복원 후 검증 loss | |

선택 epoch는 회복 학습 구간을 기준으로 기록합니다.

검증 정확도가 가장 높은 모델을 선택하며, 동률이면 검증 loss가 낮은 모델을 선택합니다. 테스트 데이터는 모델 선택에 사용하지 않습니다.

## 회복 학습 설정

- Batch size: 8
- Optimizer: AdamW
- Backbone learning rate: `1e-5`
- Head learning rate: `1e-4`
- Weight decay: `0.01`
- Scheduler: 사용하지 않음
- 학습: AMP float16
- 검증: FP32
- Pruning 이후 optimizer를 새로 생성합니다.

## 관련 파일

- `config.json`: 회복 학습 설정, 시작 체크포인트와 실행 환경
- `history.json`: 회복 학습의 epoch별 학습과 검증 기록
- `pruning_spec.json`: 축소된 구조와 유지한 채널 정보
- `notebooks/teacher_B30_prepare_from_B0_colab.ipynb`: B-0 복원, 재평가, pruning과 초기 파일 저장
- `notebooks/teacher_B30_recovery_from_B0_colab.ipynb`: B30 복원과 회복 학습

## 복원 방법

DINOv3 ConvNeXt-Base 구조에 200클래스 분류층을 구성합니다.

`pruning_spec.json`에 맞춰 각 블록의 `pwconv1`과 `pwconv2` 내부 채널 수를 변경한 뒤, `best.pth` 안의 `model_state_dict`를 복원합니다.

이미 pruning된 체크포인트이므로, 복원 과정에서 pruning을 다시 적용하지 않습니다. KD에는 회복 학습에서 선택된 `best.pth`를 사용합니다.

이 결과는 단일 seed의 pilot 실험이며, 최종 반복 실험 결과는 아닙니다.
