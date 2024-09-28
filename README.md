# K-Fold 교차 검증 ImageNet Sketch 분류

이 프로젝트는 Timm 라이브러리와 Albumentations를 사용하여 mageNet Sketch 모델을 학습하는 방법을 제공합니다. 5-fold 교차 검증을 통해 모델을 훈련하고, 각 fold에서 가장 성능이 좋은 모델을 저장합니다. 최종적으로 저장된 모델을 이용하여 앙상블 예측을 수행합니다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [필요한 환경](#필요한-환경)
3. [데이터 준비](#데이터-준비)
4. [학습 절차](#학습-절차)
5. [추론 및 앙상블](#추론-및-앙상블)
6. [모델 및 결과](#모델-및-결과)
7. [실행 방법](#실행-방법)

---

## 프로젝트 개요

이 프로젝트는 타임(Timm)을 사용해 다양한 사전 학습된 모델을 활용하고, Albumentations로 데이터 증강을 수행합니다. 

- 이미지 데이터를 로드합니다.
- K-Fold 교차 검증 (stratified sampling 사용)을 적용합니다.
- Timm에서 제공하는 사전 학습된 `caformer_b36.sail_in22k` 모델을 활용하여 학습과 검증을 진행합니다.
- 클래스 불균형 문제를 다루기 위해 Focal Loss를 사용합니다.
- 각 fold에서 가장 성능이 좋은 모델을 저장합니다.
- 저장된 모델을 사용하여 앙상블 예측을 수행합니다.
- 예측 결과를 CSV 파일로 저장하여 추가 평가에 활용합니다.

---

## 필요한 환경

다음과 같은 패키지와 환경이 필요합니다.
- Python
- PyTorch
- Timm
- Albumentations
- OpenCV
- Pandas
- Scikit-learn
- TQDM

아래 명령어로 필요한 패키지를 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

---
## 데이터 준비
1. 학습 데이터 구조: 학습 데이터는 data/train 폴더에 이미지 파일로 저장되며, 각 이미지에 대한 메타데이터는 train.csv 파일에 저장됩니다.
    - train.csv는 두 개의 열을 포함해야 합니다:
    - image_path: 이미지 파일 경로
    - target: 해당 이미지의 클래스 레이블

2. 추론 데이터 구조: 추론을 위한 데이터는 data/test 폴더에 이미지 파일로 저장되며, test.csv 파일에 각 이미지의 경로 정보가 저장됩니다.
데이터셋 링크
---
## 학습 절차

##### 1. 모델 설정
모델은 TimmModel 클래스를 사용하여 Timm 라이브러리에서 사전 학습된 caformer_b36.sail_in22k 모델을 불러옵니다. 이 모델은 미리 학습된 가중치를 활용하며, 출력 클래스를 설정하여 학습에 사용됩니다.

```python
model_selector = ModelSelector(
    model_type='timm', 
    num_classes=num_classes,
    model_name='caformer_b36.sail_in22k', 
    pretrained=True
)
model = model_selector.get_model().to(device)
```