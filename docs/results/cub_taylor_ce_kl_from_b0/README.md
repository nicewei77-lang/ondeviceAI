# CUB Git B-0 기반 Taylor + CE + KL 및 Student KD

GitHub B-0를 출발 모델과 frozen KL Teacher로 사용한 완료 실험이다. 2026-10-28에 저장된 완료 영수증, 학습 이력과 Test CSV를 대조해 정리했다.

## 실행 자료

- [가지치기 노트북](../../../notebooks/teacher_B30_50_70_Taylor_CE_KL_from_B0_colab.ipynb): 30%까지 40 epoch, 50%와 70%까지 각각 추가 20 epoch, 합계 80 Teacher epoch.
- [KD 노트북](../../../notebooks/kd_Taylor_CE_KL_from_B0_colab.ipynb): B-0, B-30, B-50, B-70 각각 Student seed42/43/44, 각 60 epoch, 합계 12개 작업, 720 epoch.
- [실행 코드와 복원 안내](../../../experiments/cub_taylor_kd/README.md).
- [전체 모델 다운로드 목록과 checksum](artifacts.json). 가중치는 Git에 포함하지 않는다.

## 공통 입력

CUB-200-2011의 공유 Train 4,794 / Validation 1,200 / Test 5,794 ID 목록과 순서를 유지한다. 라벨은0~199다.
Train은 RGB, HFlip0.5, 밝기0.9~1.1, pad224, ToTensor, ImageNet mean/std 순서다. Eval은 무작위 증강 없이 같은 pad와 정규화를 적용한다.
Padding은 bilinear, RGB(128,128,128), 중앙 정렬이다. Taylor calibration은 Train에서 클래스당 2장씩 400장, 고정 train augmentation epoch0, FP32 gate gradient와 FP64 누적을 사용한다.

## Teacher

Source B-0는 원래 CUB20 epoch 중 epoch10의 best.pth다. 이 실험의 SHA-256은 `3ceffe27b7d41a711937961e117a28d95b2d5611be4dc473969155b78620f207`다.
동일 B-0를 FP32 eval/no_grad로 고정하고 회복 모델에 `CE + 0.5*T²*KL(B0||pruned)`, T4를 적용한다.
학습 batch16/effective64, WD0.05, backbone/head peak LR1e-5/1e-4, stage-local warmup2와 cosine을 사용한다. 각 pruning 경계에서 AdamW/scaler를 새로 만든다.
비율은 원래 36개 block의 MLP hidden 폭 기준 누적 제거율이다. 다음 목표는 이전 목표의 LAST에서 출발하고, 각 목표의 마지막 quarter에서 Validation OA, CE, 이른 epoch 순서로 best를 선택한다.
Teacher의 B-0 확인, Validation, Test batch는8이다. B-0 원학습 recipe와 회복 recipe를 구분한다.

| Teacher | Test OA (%) | Test Hard OA (%) |
|---|---:|---:|
| B0 | 82.4991 | 63.9175 |
| B30 | 83.3103 | 63.7457 |
| B50 | 82.3438 | 63.4021 |
| B70 | 79.2889 | 60.1375 |

## Student

Student는 MobileNetV3-Large width1/dropout0.2, ImageNet1K V2다. 동일 seed의 원래 초기화를 사용한다.
Loss는 `0.5*CE + 0.5*T²*KL(Teacher||Student)`, T=4다. AdamW LR0.0003/WD0.0001, warmup 5 epoch와 cosine 60 epoch, batch16/effective64, Student eval batch16이다.
Student-only 세 모델은 같은 입력과 학습 계약의 기존 완료 모델을 재사용했다. 신규 baseline 학습은 없다. Hard 지표는 이번 B-0에서 고정한 동일 40클래스로 다시 계산했다.

| Student 조건 | Test OA, 평균 ± 표본 SD (%) | Test Hard OA, 평균 ± 표본 SD (%) |
|---|---:|---:|
| Student-only | 67.4836 ± 0.4673 | 47.2509 ± 1.9685 |
| KD B0 | 72.2184 ± 0.4143 | 51.3173 ± 1.2867 |
| KD B30 | 72.4313 ± 0.2976 | 51.4605 ± 0.9908 |
| KD B50 | 72.7649 ± 0.1990 | 51.7468 ± 0.3472 |
| KD B70 | 72.8857 ± 0.1827 | 52.0619 ± 0.6819 |

## 근거와 해석 범위

각 Test는 5,794장 전체, Hard는 1,164장이다. 제외 표본은 없다. [Teacher CSV](teacher_metrics.csv), [Student CSV](student_metrics.csv), [예측 원문](predictions/), [학습 결과 원문](evidence/), [검증 기록](validation.json)을 제공한다.
정확도는 정답수/표본수×100, CE는 표본별 정답 class 음의 로그확률의 평균이다. Hard는 true label이 고정 40클래스에 속하는 표본이며200개 전체 출력의 argmax를 사용한다.
평균과 표본 SD는 Student seed 42/43/44의 3개 결과이며 confidence interval이 아니다. 한 Teacher bank와 한 split에 조건부인 결과다.
기존에 관찰한 CUB Test를 사용했다. 새로운 holdout 또는 독립 재현으로 주장하지 않는다. Test로 checkpoint를 다시 선정하지 않았다.
이번 정리는 저장 CSV와 선택 규칙을 재계산한 것이며, 새 학습, GPU 추론, 모델 tensor 복원은 실행하지 않았다. Android latency는 측정하지 않았다.

원문: [Teacher 완료](https://drive.google.com/file/d/12wa1nr4CDFRRJHDgMyNE2WyLEYiNq-Wc/view), [Teacher Test 완료](https://drive.google.com/file/d/1i34-Wtv0ajSMn56njJgzGDcQNS6Vof7P/view), [KD 완료](https://drive.google.com/file/d/1HXlOdEKNYDT-XWN4XZM-i-IezZs8Iqet/view).
