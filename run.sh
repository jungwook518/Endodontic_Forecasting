#!/bin/bash

python main_5folds.py --exp_name "exp_0001" --learning_rate 0.1
python main_5folds.py --exp_name "exp_0002" --learning_rate 0.01
python main_5folds.py --exp_name "exp_0003" --learning_rate 0.001
python main_5folds.py --exp_name "exp_0004" --learning_rate 0.0001
python main_5folds.py --exp_name "exp_0005" --learning_rate 0.00001