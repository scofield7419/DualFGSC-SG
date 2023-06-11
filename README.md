
Codes for the SIGIR 2022 paper 
[Mutual disentanglement learning for joint fine-grained sentiment classification and controllable text generation](https://dl.acm.org/doi/abs/10.1145/3477495.3532029)


-------

## Requirements

```bash
conda create --name dual-scsg python=3.8
pip install -r requirements.txt
```


-------

## Data

Download the datasets and put them on the corresponding folds.

- Fine-grained ABSA
  - [laptop15](data%2Flaptop15)
  - [restaurant14](data%2Frestaurant14)
  - [restaurant15](data%2Frestaurant15)
  - [restaurant16](data%2Frestaurant16)
  - [mams](data%2Fmams)
- Coarse-grained ABSA
  - [trip_advisor](data%2Ftrip_advisor)
  - [beer_advocate](data%2Fbeer_advocate)


 

-------


## Usage


### Step 1. Train language model


```bash
python run_lm.py 
```


### Step 2. Train the backbone dual models

```bash
python run_dsl.py
```


### Step 3. Mutual disentanglemet learning



```bash
python main.py
```


**Note: properly configurate the parameters of each script.**




## Cite

```
@inproceedings{sigir22-dual-scsg,
  author       = {Hao Fei and
                  Chenliang Li and
                  Donghong Ji and
                  Fei Li},
  title        = {Mutual Disentanglement Learning for Joint Fine-Grained Sentiment Classification
                  and Controllable Text Generation},
  booktitle    = {Proceedings of the 45th International {ACM} {SIGIR} Conference on Research
                  and Development in Information Retrieval},
  pages        = {1555--1565},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3477495.3532029}
}
```


## License

The code is released under Apache License 2.0 for Noncommercial use only. 

