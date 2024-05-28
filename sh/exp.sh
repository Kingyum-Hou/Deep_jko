# conda activate pytorch
export PYTHONPATH="/data/jingren/repository/AI4science/JKO/my_jko"
nohup python -u wassersteinGF/main.py config_path='exp_config/wassersteinGF_KL.yaml' >> logs/temp.log 2>&1 &