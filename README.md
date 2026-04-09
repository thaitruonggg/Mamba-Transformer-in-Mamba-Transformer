# MaMa: Mamba-Transformer in Mamba-Transformer for Traffic Sign Recognition

![MaMa architecture](result_plots/3.1.png)

# Getting started

## Installation

To get started, clone the MaMa repository and navigate to the project directory:
```
git clone https://github.com/naofunyan/Mamba-Transformer-in-Mamba-Transformer.git
cd Mamba-Transformer-in-Mamba-Transformer
```
> [!CAUTION]
> We recommend running the project on Linux for easier Mamba wheel installation.

## Datasets
GTSRB and TT100K datasets can be downloaded at:

- German Traffic Sign Recognition Benchmark - GTSRB: [![GTSRB](https://img.shields.io/badge/Kaggle-GTSRB-link?style=flat&logo=kaggle&color=blue&link=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fnaofunyannn%2Fmama-gtsrb)](https://www.kaggle.com/datasets/naofunyannn/mama-gtsrb)
  
    A dataset is a directory with the following structure:
  ```bash
  $ tree data
  GTSRB
  ├── GTSRB_Final_Test_GT
  │   └── GT-final_test.csv
  ├── GTSRB_Final_Test_Images
  │   └── GTSRB
  │       ├── Final_Test
  │       │     └── GTSRB
  │       │         ├── 00000.ppm
  │       │         ├── 00001.ppm
  │       │         └── ...
  │       ├── test
  │       │     ├── 0000
  │       │     │   ├── 00243.ppm
  │       │     │   ├── 00252.ppm
  │       │     │   └── ...
  │       │     ├── 0001
  │       │     │   ├── 00001.ppm
  │       │     │   ├── 00024.ppm
  │       │     │   └── ...
  │       │     └── ...       
  │       └── Readme-Images-Final-test.txt
  └── GTSRB_Final_Training_Images
      └── GTSRB
          ├── Final_Training
          │      └── Images
          │           ├── 00000
          │           │    ├── 00000_00000.ppm
          │           │    ├── 00000_00001.ppm
          │           │    └── ...
          │           ├── 00001
          │           │   ├── 00000_00000.ppm
          │           │   ├── 00000_00001.ppm
          │           │   └── ...
          │           └── ... 
          └── Readme-Images.txt

- Tsinghua-Tencent 100K - TT100K: From the author [![Static Badge](https://img.shields.io/badge/Dataset-TT100K-blue?logo=ieee&labelColor=gray&color=green&link=https%3A%2F%2Fcg.cs.tsinghua.edu.cn%2Ftraffic-sign%2F)](https://cg.cs.tsinghua.edu.cn/traffic-sign/) or our complete package [![Static Badge](https://img.shields.io/badge/Kaggle-TT100K-blue?logo=kaggle&labelColor=gray&color=blue&link=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fnaofunyannn%2Fmama-tt100k)](https://www.kaggle.com/datasets/naofunyannn/mama-tt100k)
  
  A dataset is a directory with the following structure:
    ```bash
    $ tree data
    TT100K
    ├── marks
    │      ├── i1.png
    │      ├── i2.png
    │      └── ...
    ├── organized_test
    │      ├── i1
    │      ├── i2
    │      └── ...
    ├── organized_train
    │      ├── i1
    │      ├── i2
    │      └── ...
    ├── other
    │      ├── 23723.jpg
    │      ├── 23739.jpg
    │      └── ...
    ├── test
    │      ├── 2.jpg
    │      ├── 13.jpg
    │      └── ...
    ├── train
    │      ├── 23.jpg
    │      ├── 35.jpg
    │      └── ...
    ├── annotations_all.json
    ├── marks.jpg
    ├── report.pdf
    └── test_result.pkl

# Results
![Result1](result_plots/4.18.png)
![Result2](result_plots/4.19.png)
![Result3](result_plots/4.20.png)