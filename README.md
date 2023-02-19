# Endodontic_Forecasting
Endodontic forecasting model by analyzing preoperative dental radiographs

## STEP 0. Prepare Base Data
1. 01.Inputs/{P-1,...P-682}.bmp 데이터 ```./Data/Mask/``` 위치로 옮기기
2. 01.Inputs/{P-1,...P-682}.dcm 데이터 ```./Data/Original/``` 위치로 옮기기
3. 01.Inputs/Preprocessed/{P-1,...P-682}.bmp 데이터 ```./Data/Preprocessed/``` 위치로 옮기기
4. 위에 데이터 샘플들은 제공해준 데이터.
5. 학습 데이터를 더 만들고 싶으면 Original dcm 데이터, 노란색 박스친 Mask bmp 데이터를 추가하면 됨.
6. Orignal, Mask, Preprocessed에 속하는 파일이름은 같은 쌍끼리 같은 이름으로 작성하고 덮어쓰기 유의하기.
6. all_label.csv에 눈치껏 추가하기

## STEP 1. Prepare Preprocessed data and Crop data for Original data.
```
python preprocessing.py
```

## STEP 2. Train & Test
main.py에서 exp_name 바꿔가면서 실험하기.
```
python main.py 
```

## STEP 3. Check Model & Score & Log
each best model file save location
```
./result/exp_name/model
``` 
each best model score save location
```
./result/exp_name/score
``` 
loss and acc during training save location
```
./result/exp_name/figure
```



