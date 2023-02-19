# Endodontic_Forecasting
Endodontic forecasting model by analyzing preoperative dental radiographs

## STEP 0. Prepare Base Data
1. 01.Inputs/{P-1,...P-682}.bmp 데이터 ```./Data/Mask/``` 위치로 옮기기
2. 01.Inputs/{P-1,...P-682}.dcm 데이터 ```./Data/Original/``` 위치로 옮기기
3. 01.Inputs/Preprocessed/{P-1,...P-682}.bmp 데이터 ```./Data/Preprocessed/``` 위치로 옮기기
위에 데이터 샘플들은 제공해준 데이터.
학습 데이터를 더 만들고 싶으면 Original dcm 데이터, 노란색 박스친 Mask bmp 데이터를 추가하면 됨.
all_label.csv에 추가하기

## STEP 1. Prepare Crop data for Original data.
```
python crop.py
```

## STEP 2. Prepare Preprocessed data for Crop data.
Not Yet
```
Not Yet
```

## STEP 3. Train & Test
main.py에서 exp_name 바꿔가면서 실험하기.
```
python main.py 
```


## STEP 4. Check Model & Score & Log
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



