# FairSeg
> [**FairSeg: A Large-scale Medical Image Segmentation Dataset for Fairness Learning with Fair Error-Bound Scaling**](https://arxiv.org/pdf/2311.02189.pdf)
>
> by [Yu Tian*](https://yutianyt.com/), Min Shi*, [Yan Luo*](https://luoyan407.github.io/), Ava Kouhana, [Tobias Elze](http://www.tobias-elze.de/), and [Mengyu Wang](https://ophai.hms.harvard.edu/team/dr-wang/).
>
<img width="729" alt="Screenshot 2023-11-03 at 1 10 02 PM" src="https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg/assets/19222962/d3948ed8-1321-482f-8866-165d5a9ab2e4">


## Fairness Datasets

* :beers::beers: For more fairness datasets including 2D and 3D images of three different eye diseases, please check our dataset [**webpage**](https://ophai.hms.harvard.edu/datasets/)!

## Download FairSeg Dataset
* Our FairSeg dataset can be downloaded via this [**link**](https://drive.google.com/drive/folders/1tyhEhYHR88gFkVzLkJI4gE1BoOHoHdWZ?usp=sharing).

## Installation
Linux (We tested our codes on Ubuntu 18.04)
Anaconda
Python 3.7.11
Pytorch 1.9.1

First, please run the following commands:
```
conda create -n FairSeg python=3.7.11
conda activate FairSeg
pip install -r requirements.txt
```


## Quick start

Here are the instructions: 

## Training
We use 2 NVIDIA A100 GPUs for training.

After downloading our FairSeg Dataset, please specify the root_dir to train the SAMed and run this command.
```bash
./train.sh
```
After finishing the SAMed training, finetune the pretrained SAMed using our proposed Fair Error-Bound Scaling loss. Please specify the pretrained lora_ckpt and root_dir path. Then run the command below: 
```bash
./train_finetune.sh
```


## Testing

For testing, please specify the root_dir, attribute, path of pretrained lora checkpoint, and output_dir. Then run this command.
```bash
./test.sh
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
