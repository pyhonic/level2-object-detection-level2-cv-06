# 🍣 오마카세 
![image](https://user-images.githubusercontent.com/91659448/164386988-ddda3bd7-214c-4212-b657-c2fe42975d52.png)
- 대회 기간 : 2022.03.21 ~ 2022.04.08
- 목적 : 재활용 품목 분류를 위한 Object Detection

## ♻️ 재활용 품목 분류를 위한 Object Detection
### 🔎 배경
![image](https://user-images.githubusercontent.com/91659448/164387063-c84ae185-257c-4b90-8015-366cbe22a05d.png)

- 코로나19가 확산됨에 따라 언택트 시대가 도래하였습니다.
- 이에 발맞춰 배달 산업의 성장과 e커머스 시장의 확대되며 일회용품과 플라스틱의 사용 비율이 높아졌습니다.
- 이러한 문화는 해당 산업의 성장을 불러왔지만, "쓰레기 대란"과 "매립지 부족"과 같은 사회 문제를 낳고 있습니다.
- 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 
- 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정 받아 재활용 되지만 그렇지 않은 경우는 폐기물로 분류되어 매립 또는 소각되기 떄문입니다.
- 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 

### 💾 데이터 셋
- `전체 이미지 개수` : 9754장 (train 4883 장, test 4871 장)
- `10개 클래스` : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- `이미지 크기` : (1024, 1024)


## 🙂 멤버
| 박동훈 | 박성호 | 송민기 | 이무현 | 이주환 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/BTOCC25) | [Github](https://github.com/pyhonic) | [Github](https://github.com/alsrl8) | [Github](https://github.com/PeterLEEEEEE) | [Github](https://github.com/JHwan96)


## 📋 역할
| 멤버 | 역할 |
| :-: | :-: |
|박동훈(T3086)| EDA, 2-stage 모델 학습(DETR, HTC) |
|박성호(T3090)| EDA, utils 관련 파일 생성, 모델 학습 및 parameter 관련 실험 진행 |
|송민기(T3112)| EDA, 1stage 모델 학습(centernet), mmdetection 공부 |
|이무현(T3144)| 1stage 모델 학습(yolov5, TOOD), inference 시각화 |
|이주환(T3241)| EDA, 2stage, 새로운 모델 적용 및 확인 (focalNet, swin 등) |


## 🧪 실험
|Property|Model|Backbone|mAP@public|mAP@private|
| :-: | :-: | :-: | :-: | :-: | 
| 1-Stage | YOLOv5-L | CSPDarkNet | 0.5287 | 0.5014 | 
| 2-Stage | Faster R-CNN | Swin-L | 0.6344 | 0.6199 | 
| Ensemble | Swin-L, YOLOv5-L | | 0.6841 | 0.6680 | 

- [실험 노션](https://overjoyed-exoplanet-127.notion.site/79557585126a4f7e80deaf482566cce7?v=8bb209b39c0a4f24a4600e91380ade73)

## Reference
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [BoxInst](https://github.com/wangbo-zhao/OpenMMLab-BoxInst)
