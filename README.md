# Harvard-FairSeg
> [**ICLR'24**] [**Harvard FairSeg: A Large-Scale Medical Image Segmentation Dataset for Fairness Learning Using Segment Anything Model with Fair Error-Bound Scaling**](https://arxiv.org/pdf/2311.02189.pdf)
>
> by [Yu Tian*](https://yutianyt.com/), [Min Shi*](https://shiminxst.github.io/index.html), [Yan Luo*](https://luoyan407.github.io/), Ava Kouhana, [Tobias Elze](http://www.tobias-elze.de/), and [Mengyu Wang](https://ophai.hms.harvard.edu/team/dr-wang/).
>
<img width="1081" alt="Screenshot 2024-01-20 at 9 24 39 AM" src="https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-FairSeg/assets/19222962/176cd0d2-f3ec-4ac2-a0cb-65d66574f25b">



## Download Harvard-FairSeg Dataset
* Our Harvard-FairSeg dataset can be downloaded via this [**link**](https://ophai.hms.harvard.edu/datasets/harvard-fairseg10k/).

* Alternatively, you could also use this [**Google Drive link**](https://drive.google.com/drive/u/1/folders/1tyhEhYHR88gFkVzLkJI4gE1BoOHoHdWZ) to directly download our Harvard-FairSeg dataset.  

* Please refer to each of the folders for FairSeg with **SAMed** and **TransUNet**, respectively. 

* [CVer中文讲解](https://zhuanlan.zhihu.com/p/680169908)

  
# Dataset Description

This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

The dataset contains 10,000 patients includes 10,000 SLO fundus images. The cup-disc mask, patient age, sex, race, language, marital status, and ethnicity information are also included in the data.

10,000 SLO fundus images with pixel-wise cup-disc masks are in the Google Drive folder:
data_00001.npz
data_00002.npz
...
data_10000.npz

NPZ files have the following keys:
```    
fundus_slo: SLO fundus image
disc_cup_borders: cup-disc mask for the corresponding SLO fundus image
age: patient's age
race: 0 - Asian, 1 - Black, 2 - White
gender: 0 - Female, 1 - Male
ethnicity: 0 - Non-Hispanic, 1 - Hispanic
language: 0 - English, 1 - Spanish, 2 - Others
marriagestatus: 0 - Married, 1 - Single, 2 - Divorced, 3 - Widowed, 4 - Leg-Sep
```


## More Fairness Datasets

* :beers::beers: For more fairness datasets including 2D and 3D images of three different eye diseases, please check our dataset [**webpage**](https://ophai.hms.harvard.edu/datasets/)!

## Acknowledgement & Citation


If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{tian2024fairseg,
      title={Harvard FairSeg: A Large-Scale Medical Image Segmentation Dataset for Fairness Learning Using Segment Anything Model with Fair Error-Bound Scaling}, 
      author={Yu Tian, Min Shi, Yan Luo, Ava Kouhana, Tobias Elze, Mengyu Wang},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2024},
}
