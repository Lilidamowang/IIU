# IIU: Independent Inference Units for Knowledge-based Visual Question Answering
This is a Pytorch implementation for IIU: Independent Inference Units for Knowledge-based Visual Question Answering (KSEM 2024).

# Requirements
1. Install Python 3.7.
2. Install PyTorch 1.2.
3. Install other dependency packages.
4. Clone this repository and enter the root directory of it.
```
git clone git@github.com:Lilidamowang/IIU.git
```

# Usage
For training the model
```
CUDA_VISIBLE_DEVICES=0 python train.py --config-yml config/config_okvqa.yml --cpu-workers 8 --gpus 0 --save-dirpath checkpoints
```
* config-yml: Path to a config file listing reader, model and solver parameters.
* cpu-workers: Number of CPU workers for dataloader.
* save-dirpath: Path of directory to create checkpoint directory and save checkpoints.
* load-pthpath: To continue training, path to .pth file of saved checkpoint.
* validate: Whether to validate on val split after every epoch.
  