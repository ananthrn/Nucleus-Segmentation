To reproduce our experiments with weighted U-Net,
place the .py scripts in the same folder as the dataset
(stage1_train/; stage1_test/; stage1_solution.csv), and
run the following commands:

1. Preprocess data

python3 wunet_preprocess.py

2. Create weight maps

python3 wunet_create_weights.py

3. Train and evaluate model

python3 wunet_model.py