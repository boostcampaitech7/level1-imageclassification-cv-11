# %%

import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import pandas as pd


# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
traindata_dir = "./data/train"
traindata_info_file = "./data/train.csv"
save_result_path = "./train_result"

# 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
train_info = pd.read_csv(traindata_info_file)

# 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.

testdata_dir = "./data/test"
testdata_info_file = "./data/test.csv"
test_info = pd.read_csv(testdata_info_file)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %%


model_names = ['caformer', 'coatnet', 'effv2', 'eva14', 'nfnet', 'resnet101']

sub_list = pd.DataFrame({'target': [0]*10014})

for s in model_names:
    sub = pd.read_csv('./csv/output_'+s+'.csv')
    sub_list[s] = sub['target']

plt.figure(figsize=(8, 6))
g = sns.heatmap(sub_list.iloc[:, 1:].corr(), annot=True)
g.set_title("Correlation between models")
plt.show()

# %%

test_accuracy = {model_names[0]: 0.9110,
                 model_names[1]: 0.8940,
                 model_names[2]: 0.8750,
                 model_names[3]: 0.8880,
                 model_names[4]: 0.8890,
                 model_names[5]: 0.8880
                 }

scatters = pd.DataFrame({'correlation': sub_list.iloc[:, 1:].corr(
).iloc[0], 'accuracy': test_accuracy.values(), 'name': test_accuracy.keys()})
plt.scatter(scatters['correlation'], scatters['accuracy'],)

for i in range(len(scatters)):  # 행 개수만큼 순회
    row = scatters.iloc[i]  # 한 행씩 꺼내기
    name = row['name']  # 이름이 저장된 열
    x = row['correlation']  # x좌표가 저장된 열
    y = row['accuracy']  # y좌표가 저장된 열

    plt.text(x+0.001, y-0.001, name)  # x 좌표, y좌표, 이름 순서로 input 지정

plt.show()

# %%

sub_list2 = sub_list.copy()
sub_list.drop(['effv2'], axis=1).iloc[:, 1:]

# %%
# sub_list

final_prediction = sub_list.drop(['effv2'], axis=1).iloc[:, 1:].mode(axis=1).iloc[:, 0].astype(int)
final_prediction

# %%
# 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
test_info = pd.read_csv(testdata_info_file)
test_info['target'] = final_prediction
test_info = test_info.reset_index().rename(columns={"index": "ID"})
test_info

# DataFrame 저장
test_info.to_csv("../submissions/god_please.csv", index=False)


pd.read_csv('../submissions/god_please.csv')
