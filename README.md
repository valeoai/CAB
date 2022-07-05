# Introduction

This repository is the code for our article [Raising context awareness in motion forecasting](https://arxiv.org/abs/2109.08048), accepted at CVPR Workshop WAD 2022. It was built based on the [Trajectron ++ repository](https://github.com/StanfordASL/Trajectron-plus-plus).

# Setup 

Start by downloading the [nuScenes dataset](https://www.nuscenes.org/) (v1.0) along with the map expansion pack (v1.1). Then, install the requirements

```
pip install -r requirements.txt
```

# Pre-processing

The preprocessing code is similar to the original one, excepts that it splits the data according to the Challenge split (see [this issue](https://github.com/StanfordASL/Trajectron-plus-plus/issues/11)). To pre-process the data, please run

```
cd CAB/trajectron && 
python process_data_challenge.py --data /path/to/nuscenes/ --version "v1.0-trainval" --output_path /desired/preprocessed/location
```

This script will create the following `.pkl` files:
```
nuScenes_train_full.pkl  
nuScenes_train_val_full.pkl  
nuScenes_val_full.pkl
```

# Train models

Here are the commands you can use to train the models presented in our article. Before training, please specify in the `data_dir` field of the config files the location of the pre-processed dataset (the value of `--output_path` in the `process_data_challenge.py` script). 

### CAB

```
cd CAB/trajectron &&
python train.py -o ../experiments/nuScenes/models/cab/config.yaml --log_dir /desired/log/dir
```

### RubiZ

```
cd CAB/trajectron &&
python train.py -o ../experiments/nuScenes/models/rubiz/config.yaml --log_dir /desired/log/dir
```


### Reweight

```
cd CAB/trajectron &&
python train.py -o ../experiments/nuScenes/models/reweight/config.yaml --log_dir /desired/log/dir
```

### Trajectron

```
cd CAB/trajectron &&
python train.py -o ../experiments/nuScenes/models/trajectron/config.yaml --log_dir /desired/log/dir
```

### Trajectron (no-context)

```
cd CAB/trajectron &&
python train.py -o ../experiments/nuScenes/models/trajectron_nc/config.yaml --log_dir /desired/log/dir
```

### HalentNet

```
cd CAB/trajectron &&
python train_halentnet.py -o ../experiments/nuScenes/models/halentnet/config.yaml --log_dir /desired/log/dir
```

# Evaluate models 

Model evaluation is done using two scripts. The first one computes all the predictions on the given dataset.

```
MODEL_DIR="/directory/of/saved/experiment" && CKPT=epoch_you_want_to_evaluate
python evaluate_new.py -o $MODEL_DIR/config.yaml --resume $CKPT  --eval_data_dict nuScenes_val_full.pkl
```

This will create a folder in your logs that contains all the predictions. Then, you can use these predictions to computes the metrics values:

```
python compute_val_metrics.py --model_dir $MODEL_DIR --checkpoint $CKPT
```

Metrics values are prompted and saved in `$MODEL_DIR/metrics/epoch,$CKPT/val/metrics.pkl`

# Citation

If you use our code and/or our article, you can cite us using:

```
@InProceedings{cab2022,
    author    = {Ben-Younes, H\'edi and Zablocki, \'Eloi and Chen, Micka\"el and P\'erez, Patrick and Cord, Matthieu},
    title     = {Raising Context Awareness in Motion Forecasting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {4409-4418}
}
```