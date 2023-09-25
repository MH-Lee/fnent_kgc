# CUDA_VISIBLE_DEVICES=1 python -u main.py --load_config --config_path train_yaml/dualattne_wn18rr.yaml

# CUDA_VISIBLE_DEVICES=0 python -u main.py --load_config --config_path train_yaml/dualattne_nell995.yaml > ./logs/dualattne_nell995_230922_fnet_dim512.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -u main.py --load_config --config_path train_yaml/ConvE_nell995.yaml > ./logs/conve_nell995_230922.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -u main.py --load_config --config_path train_yaml/rotate_nell995.yaml > ./logs/rotate_nell995_230922.log 2>&1 &
