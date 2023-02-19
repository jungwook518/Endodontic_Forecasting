# Endodontic_Forecasting
Endodontic forecasting model by analyzing preoperative dental radiographs

## STEP 0. Prepare Base Data
01.Inputs/{P-1,...P-682}.bmp 데이터 ```./Data/Mask/``` 위치로 옮기기
01.Inputs/{P-1,...P-682}.dcm 데이터 ```./Data/Original/``` 위치로 옮기기
01.Inputs/Preprocessed/{P-1,...P-682}.dcm 데이터 ```./Data/Preprocessed/``` 위치로 옮기기

## STEP 1. Crop Original data.
```python crop.py ```

## STEP 2. Train & Test
```python main.py ```
main.py에서 exp_name 바꿔가면서 실험하기.

## STEP 3. Check Model & Score & Log
```./result/exp_name/model``` each best model file save location
```./result/exp_name/score``` each best model score save location
```./result/exp_name/figure```loss and acc during training save location



