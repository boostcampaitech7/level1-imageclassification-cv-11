import os
from typing import Tuple, Callable, Union
import cv2
import timm
import torch
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        # 데이터셋 기본 경로, 이미지 변환 방법, 이미지 경로 및 레이블 초기화
        self.root_dir = root_dir  # 이미지 파일 root
        self.transform = transform  # 이미지 변환 처리
        self.is_inference = is_inference # 추론 확인
        self.image_paths = info_df['image_path'].tolist()  # 이미지 파일 경로 목록
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()  # 각 이미지에 대한 레이블 목록

    def __len__(self) -> int:
        # 데이터셋의 총 이미지 수를 반환
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        # 주어진 인덱스에 해당하는 이미지를 로드하고 변환을 적용한 후, 이미지와 레이블을 반환
        img_path = os.path.join(self.root_dir, self.image_paths[index])  # 이미지 경로 조합  
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 Read
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환
        image = self.transform(image)  # 설정된 이미지 변환을 적용

        if self.is_inference:
            return image
        else:
            target = self.targets[index]  # 해당 이미지의 레이블
            return image, target  # 변환된 이미지와 레이블을 튜플 형태로 반환
        
class UnsharpMask(A.ImageOnlyTransform):
    def __init__(self, kernel_size, sigma, amount, threshold, always_apply=False, p=1.0):
        super(UnsharpMask, self).__init__(always_apply, p)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold

    def apply(self, image, **params):
        return self.unsharp_mask(image)

    def unsharp_mask(self, image):
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        sharpened = cv2.addWeighted(image, 1.0 + self.amount, blurred, -self.amount, 0)
        if self.threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < self.threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 그레이스케일 변환, 언샤프 마스크, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            UnsharpMask(kernel_size=7, sigma=1.5, amount=1.5, threshold=0, p=1.0),  # UnsharpMask 적용
            A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.2), contrast_limit=0, p=1.0), # 20% 어둡게
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화            
            ToTensorV2(),  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        if is_train:
            # 훈련용 변환: 랜덤 크롭, 랜덤 수평 뒤집기, 랜덤 회전 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=30),  # 최대 30도 회전
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5), # 50% 확률로 블럭 추가
                    A.GridDistortion(num_steps=1, distort_limit=(-0.03, 0.05), interpolation=2, border_mode=0, value=(255, 255, 255), p=1) # 격자모양으로 왜곡을 적용
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed['image']  # 변환된 이미지의 텐서를 반환
    
class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str):
        # 지원하는 변환 라이브러리인지 확인
        if transform_type == "albumentations":
            self.transform_type = transform_type        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):        
        transform = AlbumentationsTransform(is_train=is_train)
                
        return transform
    
class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
    
class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):        
        # 모델 유형에 따라 적절한 모델 객체를 생성        
        if model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model
    
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # Cross Entropy Loss 계산
        pt = torch.exp(-ce_loss) # 예측 확률 계산
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss # Focal Loss 계산

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        
class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler  # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = []  # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf')  # 가장 낮은 Loss를 저장할 변수

    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
                
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)  # 예측된 클래스
            correct_predictions += (preds == targets).sum().item()  # 맞춘 예측 개수
            total_samples += targets.size(0)  # 전체 샘플 수
            progress_bar.set_postfix(loss=loss.item())
            
        accuracy = correct_predictions / total_samples * 100  # 정확도 계산
        print(f"Train Accuracy: {accuracy:.2f}%")            

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # 정확도 계산
                _, preds = torch.max(outputs, 1)  # 예측된 클래스
                correct_predictions += (preds == targets).sum().item()  # 맞춘 예측 개수
                total_samples += targets.size(0)  # 전체 샘플 수                
                progress_bar.set_postfix(loss=loss.item())

        accuracy = correct_predictions / total_samples * 100  # 정확도 계산
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        return total_loss / len(self.val_loader)

    def train(self) -> float:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step()

        # 학습 완료 후 최종 검증 손실 반환
        return self.lowest_loss
    
# 학습에 사용할 장비를 선택.
# torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
traindata_dir = "./data/train"
traindata_info_file = "./data/train.csv"
save_result_path = "./train_result_v4_final"

# 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
train_info = pd.read_csv(traindata_info_file)

# 총 class의 수를 측정.
num_classes = len(train_info['target'].unique())

# KFold 설정
n_splits = 5  # 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []

# 학습에 사용할 Loss를 선언.
loss_fn = FocalLoss(alpha=0.5, gamma=2) # focal_loss

# KFold 교차 검증 수행
for fold, (train_idx, val_idx) in enumerate(skf.split(train_info, train_info['target'])):
    print(f'Fold {fold + 1}/{n_splits}')
    
    # 학습에 사용할 Transform을 선언
    transform_selector = TransformSelector(transform_type = "albumentations")
    
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # train_df와 val_df를 train_idx와 val_idx로 분할
    train_df = train_info.iloc[train_idx]
    val_df = train_info.iloc[val_idx]
    
    # 학습에 사용할 Dataset 선언
    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )

    # 학습에 사용할 DataLoader 선언
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 학습에 사용할 Model 선언 (매 Fold마다 모델을 초기화)
    model_selector = ModelSelector(
        model_type='timm', 
        num_classes=num_classes,
        model_name='caformer_b36.sail_in22k', 
        pretrained=True
    )
    model = model_selector.get_model().to(device)

    # optimizer 선언
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-5)

    # 스케줄러 선언
    steps_per_epoch = len(train_loader)  # 1epoch step
    scheduler_gamma = 0.5  # 학습률을 현재의 50%로 감소
    epochs_per_lr_decay = 10  # 10 epoch마다 학습률을 감소
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step_size, 
        gamma=scheduler_gamma
    )

    # Trainer 선언
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=40,  # 각 fold마다 동일한 epoch수로 학습
        result_path=f"{save_result_path}/fold_{fold + 1}"  # 각 fold 결과 저장
    )

    # 모델 학습 및 결과 저장
    fold_result = trainer.train()
    fold_results.append(fold_result)

# 각 Fold의 결과를 기반으로 평균 성능 계산
average_performance = sum(fold_results) / len(fold_results)
print(f'KFold 평균 성능: {average_performance}')

# 폴드 수 및 모델 저장 경로 설정
n_folds = 5
fold_model_paths = [f"./train_result/fold_{fold + 1}/best_model.pt" for fold in range(n_folds) if fold<2]

# 저장된 모델을 불러와서 앙상블을 수행하는 함수
def ensemble_predict_folds(
    fold_model_paths: list, 
    device: torch.device, 
    test_loader: DataLoader
    ):
    models = []
    
    # 각 폴드의 베스트 모델 불러오기
    for fold_path in fold_model_paths:
        # 모델 초기화 및 로드
        model = ModelSelector(
            model_type='timm', 
            num_classes=num_classes,
            model_name='caformer_b36.sail_in22k', 
            pretrained=False
        ).get_model().to(device)
        model.load_state_dict(torch.load(fold_path, map_location=device))
        model.eval()
        models.append(model)
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            # 이미지를 GPU 또는 CPU로 이동
            images = images.to(device)
            
            # 폴드별 예측 수행
            fold_preds = []
            for model in models:
                logits = model(images)
                logits = F.softmax(logits, dim=1)  # 확률값으로 변환
                fold_preds.append(logits)
            
            # 폴드별 예측 결과 평균
            avg_preds = torch.mean(torch.stack(fold_preds), dim=0)
            final_preds = avg_preds.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(final_preds.cpu().detach().numpy())
    
    return predictions

# 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
testdata_dir = "./data/test"
testdata_info_file = "./data/test.csv"
save_result_path = "./train_result"

# 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
test_info = pd.read_csv(testdata_info_file)

# 총 class 수.
num_classes = 500

# 추론에 사용할 Transform을 선언.
transform_selector = TransformSelector(
    transform_type = "albumentations"
)
test_transform = transform_selector.get_transform(is_train=False)

# 추론에 사용할 Dataset을 선언.
test_dataset = CustomDataset(
    root_dir=testdata_dir,
    info_df=test_info,
    transform=test_transform,
    is_inference=True
)

# 추론에 사용할 DataLoader를 선언.
test_loader = DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False,
)

# 폴드별 저장된 모델을 사용한 앙상블 추론 실행
predictions = ensemble_predict_folds(
    fold_model_paths=fold_model_paths, 
    device=device, 
    test_loader=test_loader
)

# 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
test_info['target'] = predictions
test_info = test_info.reset_index().rename(columns={"index": "ID"})
test_info

# 예측 결과를 CSV 파일로 저장
test_info.to_csv("result.csv", index=False)
print(f"추론 결과가 result.csv 파일로 저장되었습니다.")