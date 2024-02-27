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

* Our FairSeg dataset can be downloaded via this [**link**](https://drive.google.com/drive/folders/1tyhEhYHR88gFkVzLkJI4gE1BoOHoHdWZ?usp=sharing).


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
@inproceedings{tian2024fairseg,
      title={FairSeg: A Large-Scale Medical Image Segmentation Dataset for Fairness Learning Using Segment Anything Model with Fair Error-Bound Scaling}, 
      author={Yu Tian, Min Shi, Yan Luo, Ava Kouhana, Tobias Elze, Mengyu Wang},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2024},
}
