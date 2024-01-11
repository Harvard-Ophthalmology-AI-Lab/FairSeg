# FairSeg-TransUNet
This repo holds code for TransUNet version of FairSeg

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on fairseg dataset. 

```bash
./train.sh
```
- Then finetune the pretrained model with FEBS loss. 

```bash
./train_febs.sh
```

- Run the test script on fairseg dataset.

```bash
python test.py
```


## Acknowledgement & Citation


If you find this repo useful for your research, please consider citing our paper:

```bibtex
@misc{tian2023fairseg,
      title={FairSeg: A Large-scale Medical Image Segmentation Dataset for Fairness Learning with Fair Error-Bound Scaling}, 
      author={Yu Tian, Min Shi, Yan Luo, Ava Kouhana, Tobias Elze, Mengyu Wang}
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

