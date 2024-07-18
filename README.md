# FairSeg
> [**ICLR 24**] [**FairSeg: A Large-Scale Medical Image Segmentation Dataset for Fairness Learning Using Segment Anything Model with Fair Error-Bound Scaling**](https://arxiv.org/pdf/2311.02189.pdf)
>
> by [Yu Tian*](https://yutianyt.com/), [Min Shi*](https://shiminxst.github.io/index.html), [Yan Luo*](https://luoyan407.github.io/), Ava Kouhana, [Tobias Elze](http://www.tobias-elze.de/), and [Mengyu Wang](https://ophai.hms.harvard.edu/team/dr-wang/).
>
<img width="1081" alt="Screenshot 2024-01-20 at 9 24 39 AM" src="https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-FairSeg/assets/19222962/176cd0d2-f3ec-4ac2-a0cb-65d66574f25b">

Note that, the modifier word “Harvard” only indicates that our dataset is from the Department of Ophthalmology of Harvard Medical School and does not imply an endorsement, sponsorship, or assumption of responsibility by either Harvard University or Harvard Medical School as a legal identity.

## Download Harvard-FairSeg Dataset
* Our Harvard-FairSeg dataset can refer to the dataset page of our lab website via this [**link**](https://ophai.hms.harvard.edu/datasets/harvard-fairseg10k/).

* Alternatively, you could also use this [**Google Drive link**](https://drive.google.com/drive/u/1/folders/1tyhEhYHR88gFkVzLkJI4gE1BoOHoHdWZ) to directly download our Harvard-FairSeg dataset.

* If you cannot directly download the Harvard-FairSeg dataset, please request access in the above Google Drive link, we will make sure to grant you access within 3-5 days. 

* Please refer to each of the folders for FairSeg with **SAMed** and **TransUNet**, respectively. 

* [CVer中文讲解](https://zhuanlan.zhihu.com/p/680169908)

  
# Dataset Description

This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

The dataset containing 10,000 patients includes 10,000 Scanning laser ophthalmoscopy (SLO) fundus images. The disc and cup masks, patient age, gender, race, ethnicity, language, and marital status information are also included in the data. Under the folder "ReadMe", the file "data_summary.csv" provides an overview of our data.

10,000 SLO fundus images with pixel-wise disc and cup masks are in the Google Drive folder:
data_00001.npz
data_00002.npz
...
data_10000.npz

NPZ files have the following keys:
```    
slo_fundus: Scanning laser ophthalmoscopy (SLO) fundus image
disc_cup_mask: disc and cup masks for the corresponding SLO fundus image
age: patient's age
gender: 0 - Female, 1 - Male
race: 0 - Asian, 1 - Black, 2 - White
ethnicity: 0 - Non-Hispanic, 1 - Hispanic, -1 - Unknown
language: 0 - English, 1 - Spanish, 2 - Others, -1 - Unknown
maritalstatus: 0 - Married or Partnered, 1 - Single, 2 - Divorced, 3 - Widowed, 4 - Legally Separated, -1 - Unknown
```


## More Fairness Datasets

* :beers::beers: For more fairness datasets including 2D and 3D images of three different eye diseases, please check our dataset [**webpage**](https://ophai.hms.harvard.edu/datasets/)!

## Acknowledgement & Citation


If you find this repository useful for your research, please consider citing our [paper](https://arxiv.org/pdf/2311.02189):

```bibtex
@inproceedings{tianfairseg,
  title={FairSeg: A Large-Scale Medical Image Segmentation Dataset for Fairness Learning Using Segment Anything Model with Fair Error-Bound Scaling},
  author={Tian, Yu and Shi, Min and Luo, Yan and Kouhana, Ava and Elze, Tobias and Wang, Mengyu},
  booktitle={The Twelfth International Conference on Learning Representations}
}
