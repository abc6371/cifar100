# ConvNeXtV2 Stage3/4 Adaptive Fusion 분류기 (CIFAR-100)

ConvNeXtV2(`convnextv2_tiny`) backbone의 stage3/stage4 feature를 게이팅(gating)으로 적응적 융합해 분류하는 모델을 CIFAR-100으로 학습/평가하는 스크립트입니다.

## 모델 구조 (`AdaptiveStage34FusionConvNeXtV2`)

- `timm`의 `convnextv2_tiny.fcmae_ft_in22k_in1k`를 `features_only=True`로 불러와 stage3, stage4 feature map 추출
- 각 stage feature는 avg-pool + max-pool을 concat한 뒤 Linear projection으로 동일한 차원(`fusion_dim`)으로 정렬
- 두 stage의 projection 결과를 concat해 작은 gate network에 통과시켜 stage별 가중치(softmax)를 산출하고, 가중합으로 최종 feature를 융합
- 융합 feature는 bottleneck classifier(Linear → GELU → Linear)로 분류
- 학습 시에는 stage3/stage4 각각에 보조 분류기(aux classifier)를 달아 auxiliary loss로 중간 feature 품질을 함께 학습 (`aux_loss_weight`)

## 학습 기법

- **Mixup / CutMix** 확률적 혼합 (40% / 60%)
- **EMA(Exponential Moving Average)** 모델을 별도로 유지, 평가는 EMA 모델로 수행
- **Layer-wise LR**: backbone은 낮은 LR(`lr * 0.1`), projection/gate는 `lr * 0.5`, classifier는 `lr` 그대로 적용
- **Warmup(Linear) + CosineAnnealing** 스케줄러
- CIFAR-100 `train=False` 셋은 학습에 전혀 사용하지 않고 평가 전용으로만 분리 (`evaluate()` 함수에서만 참조)
- `report_parameter_count()`로 파라미터 수를 상한(`max_params`)과 함께 검증

## 실행

```bash
# 신규 학습
python train_v2_final.py -new

# 체크포인트 이어서 학습 (지정 안 하면 log/model 안 최신 체크포인트 자동 탐색)
python train_v2_final.py

# 특정 체크포인트에서 재개
python train_v2_final.py -resume path/to/checkpoint.pth

# 커스텀 이미지 디렉토리 추론
python train_v2_final.py -test path/to/best.pth
```


## 결과

| 실험 | Pretrained | LR | Best Test Acc |
|---|---|---|---|
| [실험명 입력] | [True/False] | [값] | [값]% |

## 환경

```
torch
torchvision
timm
pyyaml
tqdm
Pillow
```
