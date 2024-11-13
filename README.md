# 2024 PKU NLP-DL Assignment 2

### Introduction of the structure of the project

- Files in **data/** folder are pre-downloaded and pre-processed datasets, which can be used directly via other parts of the project. If you want to re-download data from authoriy, please refer to **dataHelper.py** to modify the format of the datasets or instead modify the logics of **dataHelper.py**.

- **dataHelper.py** is responsible for preparing datasets for further usage, which corresponds to task 1.

- **train.py** is the primary python script for fine-tuning PLMs, including both full-parameter fine-tuning and PEFT. You can refer to **run.sh**, which is a bash script for automatically running large scale experiments, for a demonstration of how to use the training script. These two files corresponds to task 2.

- **RoBERTa_Adapter.py** implements a class for RoBERTa model with bottleneck adapters inserted by inheriting the RoBERTa model class provided in _transformers_ package. **Attention**: I implement a separate script **adapter_train.py** for training this model. It is different from the PEFT implemented in **train.py**, where I actually utilize _adapters_ package to conduct experiments. Hence the experimental results of PEFT in the report are NOT from the PEFT methods I implement in **RoBERTa_Adapter.py**. **RoBERTa_Adapter.py** and **adapter_train.py** corresponds to task 3.

- Extra files include (1) **requirements.txt**: requirements for building pyhton environment; (2) **NLP-DL-Assignment-2-Report.pdf**: the report for submission; (3) **.gitignore**: a git ignore file