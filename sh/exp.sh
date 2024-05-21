conda activate pytorch
nohup python -u main.py config_path='exp_config/wassersteinGF_KL.yaml' >> logs/temp.log 2>&1 &