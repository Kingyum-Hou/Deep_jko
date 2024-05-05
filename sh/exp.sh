conda activate pytorch
nohup python -u main.py config_path='exp_config/exp.yaml' >> logs/temp.log 2>&1 &