## 모델 Description

1-1. Mask R-CNN (미꾸리, 먹이생물 객체인식)
- 이미지 내에 존재하는 미꾸리 객체를 인식함과 동시에 각각의 경우(Instance)를 픽셀 단위로 분류하는 Segmentation Model
- Bbox Regression Branch에서 RoI의 특성을 입력받아 객체의 Bounding Box를 미세하기 보정함으로써 정확하게 먹이생물에 대한 Object Detection 수행

1-2. LSTM (성장도(체장, GSI) 시계열 예측)
- Long Short-Term Memory의 약자로 순차적인 데이터를 다루기 위한 알고리즘
- RNN의 단점인 긴 시퀀스에서 정보가 사라지는 문제를 해결 가능

## 모델 아키텍쳐

1-1. Mask R-CNN (미꾸리, 먹이생물 객체인식)
- Segmentation Task에는 픽셀 단위로 물체를 분류해야 하기 때문에 Mask Branch를 사용하여 작은 FCN(Fully Convolutional Network)를 거쳐 각 ROI에 대해 Binary Mask를 획득
- Faster R-CNN의 RPN에서 얻은 RoI(Region of Interest)에 대하여 객체의 class를 예측하는 classification branch, bbox regression을 수행하는 bbox regression branch를 예측

1-2. LSTM (성장도(체장, GSI) 시계열 예측)
- RNN(Recurrent Neural Network)의 한 종류
- 은닉층의 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억해야할 것들을 정합

## Input

1-1. 미꾸리, 먹이생물 객체인식
- 입력 데이터: JPG(이미지)
- 데이터를 로드하면서 자동으로 이미지 해상도 1024*1024 으로 변경

1-2. 성장도(체장, GSI) 시계열 예측
- 입력 데이터: CSV(시계열 데이터)
- input data shape (체장): (batch_size, 20, 8)
- input data shape (GSI): (batch_size, 20, 9)

## Output

1-1. 미꾸리, 먹이생물 객체인식
- output data shape (mask): (1024, 1024, 예측개수)
- output data shape (bbox): (예측개수, 4)

1-2. 성장도(체장, GSI) 시계열 예측
- output data shape (체장): (batch_size, 120)
- output data shape (GSI): (batch_size, 80)

## AI Task

1-1. 미꾸리, 먹이생물 객체인식
- (미꾸리)종자생산 객체인식(Instance Segmentation)
- (미꾸리)중간양성 객체인식(Instance Segmentation)
- (미꾸리)어미성숙 객체인식(Instance Segmentation)
- 먹이생물 객체인식(Object Detection)

1-2. 성장도(체장, GSI) 시계열 예측(Time-Series Forecasting)
- 종자생산-중간양성 성장도(체장) 시계열 예측
- 어미성숙 성장도(GSI) 시계열 예측

## Training Dataset

1-1. 미꾸리, 먹이생물 객체인식
- JPG(이미지)
- JSON(라벨링 종류 : polygon(미꾸리) / bbox(먹이생물))

1-2. 성장도(체장, GSI) 시계열 예측
- CSV(시계열 데이터)
- 성장도(체장) 예측 데이터 항목(8개): water_temp, water_do, water_orp, tank_lux, maturation_period, body_length, feed_type, feed_frequency
- 성장도(GSI) 예측 데이터 항목(9개): water_temp, water_do, water_orp, tank_lux, maturation_period, photoperiod, body_weight, gonads_weight, gsi

## Training 요소

1-1. 미꾸리, 먹이생물 객체인식 학습 파라미터(Parameter)
- (미꾸리-종자생산) epoch 250, batch 1
- (미꾸리-중간양성) epoch 200, batch 1
- (미꾸리-어미성숙) epoch 200, batch 1
- (먹이생물) epoch 50, batch 1

1-2. 성장도(체장, GSI) 시계열 예측
- (종자생산-중간양성 체장) epoch 20, batch 256, optimizer RMSProp, loss MAE
- (어미성숙 GSI) epoch 100, batch 16, optimizer RMSProp, loss MAE

## Evaluation Metric

1-1. 미꾸리, 먹이생물 객체인식
- (미꾸리-종자생산) 유효성 검증 목표수치 mAP 60% 이상, mAP 76.08% 달성
- (미꾸리-중간양성) 유효성 검증 목표수치 mAP 65% 이상, mAP 76.16% 달성
- (미꾸리-어미성숙) 유효성 검증 목표수치 mAP 65% 이상, mAP 80.91% 달성
- (먹이생물) 유효성 검증 목표수치 mAP 67.3% 이상, mAP 68.5% 달성

1-2. 성장도(체장, GSI) 시계열 예측
- (종자생산-중간양성 체장) 유효성 검증 목표수치 MAE 0.95 이하, MAE 0.68 달성
- (어미성숙 GSI) 유효성 검증 목표수치 MAE 0.95 이하, MAE 0.87 달성