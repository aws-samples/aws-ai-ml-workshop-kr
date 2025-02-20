#########################################################
# 1. Loading Python Libary
#########################################################
print("## Starting 1. Loading Python Libary ..... ")

import torch
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from datasets import Dataset, load_dataset
from datasets import load_dataset
from transformers import pipeline, set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

#########################################################
# 2. 데이터셋 다운로드
#########################################################
print("## Starting 2. 데이터셋 다운로드 ..... ")

huggingface_dataset_name = "daekeun-ml/naver-news-summarization-ko"

dataset = load_dataset(huggingface_dataset_name)
print("dataset: ", dataset)
print("dataset['train']: ", dataset['train'][0])

#########################################################
# 3. 데이터셋 변형
#########################################################
print("## Starting 3. 데이터셋 변형 ..... ")

import json

# Chat Message 형태 템플릿 정의
def format_instruction(system_prompt: str, article: str, summary: str):
    message = [
            {
                'content': system_prompt,
                'role': 'system'
            },
            {
                'content': f'Please summarize the goals for journalist in this text:\n\n{article}',
                'role': 'user'
            },
            {
                'content': f'{summary}',
                'role': 'assistant'
            }
        ]
    
    return message # json.dumps(message, indent=2) # json.dumps(message, ensure_ascii=False, indent=2)


# 사용 예시
# system_prompt = "You are an AI assistant specialized in news articles. Your role is to provide accurate summaries and insights. Please analyze the given text and provide concise, informative summaries that highlight the key goals and findings."
# article = "Within three days, the intertwined cup nest of grasses was complete, featuring a canopy of overhanging grasses to conceal it. And decades later, it served as Rinkert's portal to the past inside the California Academy of Sciences. Information gleaned from such nests, woven long ago from species in plant communities called transitional habitat, could help restore the shoreline in the future. Transitional habitat has nearly disappeared from the San Francisco Bay, and scientists need a clearer picture of its original species composition—which was never properly documented. With that insight, conservation research groups like the San Francisco Bay Bird Observatory can help guide best practices when restoring the native habitat that has long served as critical refuge for imperiled birds and animals as adjacent marshes flood more with rising sea levels. \"We can't ask restoration ecologists to plant nonnative species or to just take their best guess and throw things out there,\" says Rinkert."
# summary = "Scientists are studying nests hoping to learn about transitional habitats that could help restore the shoreline of San Francisco Bay."

# print(format_instruction(system_prompt, article, summary))

# Add system message to each conversation
columns_to_remove = list(dataset["train"].features)
print("columns_to_remove: ", columns_to_remove)

def generate_instruction_dataset(data_point):
    system_prompt = "You are an AI assistant specialized in news articles.Your role is to provide accurate summaries and insights in Korean. Please analyze the given text and provide concise, informative summaries that highlight the key goals and findings."

    return {
        "messages": format_instruction(system_prompt, data_point["document"],data_point["summary"])
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_instruction_dataset).remove_columns(columns_to_remove)
    )    
    

def create_message_dataset(dataset, num_train,num_val, num_test, verbose=False):
    ## APPLYING PREPROCESSING ON WHOLE DATASET

    train_dataset = process_dataset(dataset["train"].select(range(num_train)))
    validation_dataset = process_dataset(dataset["validation"].select(range(num_val)))
    test_dataset= process_dataset(dataset["test"].select(range(num_test)))
    
    if verbose:
        print(train_dataset)
        print(test_dataset)
        print(validation_dataset)

    return train_dataset,test_dataset,validation_dataset    

# 전체 데이터 셋에서 일부 데티터 추출 (짧은 실습을 위해서)
train_num_debug = 10
validation_num_debug = 10
test_num_debug = 10

train_dataset,test_dataset,validation_dataset = create_message_dataset(dataset=dataset, 
                                                num_train=train_num_debug,
                                                num_val=validation_num_debug, 
                                                num_test=test_num_debug, verbose=True)   

# 전체 데이터 셋 저장 (성능 측정을 위해서)    
full_train_num = len(dataset["train"])
full_validation_num = len(dataset["validation"])
full_test_num = len(dataset["test"])

# full_train_num = 1000
# full_validation_num = 1000
# full_test_num = 1000

print("train_num_samples: ", full_train_num)
print("validation_num_samples: ", full_validation_num)
print("test_num_samples: ", full_test_num)

full_train_dataset,full_test_dataset,full_validation_dataset = create_message_dataset(dataset=dataset, 
                                                                num_train=full_train_num,
                                                                num_val=full_validation_num, 
                                                                num_test=full_test_num, verbose=True)  


#########################################################
# 4. 데이터 셋을 JSON 으로 저장
#########################################################
print("## Starting 4. 데이터 셋을 JSON 으로 저장 ..... ")

import os

def create_dataset_json_file(huggingface_dataset_name,train_dataset, validation_dataset, test_dataset, is_full, verbose=True ):
    dataset_name = huggingface_dataset_name.split("/")[1]
    data_folder = os.path.join("../data/",dataset_name)
    os.makedirs(data_folder, exist_ok=True)

    if is_full:
        train_data_json = os.path.join(data_folder,"full_train", "train_dataset.json")
        validation_data_json = os.path.join(data_folder,"full_validation", "validation_dataset.json")
        test_data_json = os.path.join(data_folder, "full_test", "test_dataset.json")

    else:
        train_data_json = os.path.join(data_folder,"train", "train_dataset.json")
        validation_data_json = os.path.join(data_folder,"validation", "validation_dataset.json")
        test_data_json = os.path.join(data_folder, "test", "test_dataset.json")

    # save datasets to disk 
    train_dataset.to_json(train_data_json, orient="records", force_ascii=False)
    validation_dataset.to_json(validation_data_json, orient="records", force_ascii=False)
    test_dataset.to_json(test_data_json, orient="records", force_ascii=False)        

    if verbose:
        print(train_dataset)
        print(f"{train_data_json} is saved")
        print(f"{validation_data_json} is saved")
        print(f"{test_data_json} is saved")                

    return data_folder, train_data_json, validation_data_json, test_data_json


# Store debug dataset
data_folder, train_data_json, validation_data_json, test_data_json = create_dataset_json_file(huggingface_dataset_name=huggingface_dataset_name,
                                                                    train_dataset=train_dataset, 
                                                                    validation_dataset=validation_dataset, 
                                                                    test_dataset=test_dataset,
                                                                    is_full=False )        

# Store full dataset
data_folder, full_train_data_json, full_validation_data_json, full_test_data_json = create_dataset_json_file(huggingface_dataset_name=huggingface_dataset_name,
                                                                    train_dataset=full_train_dataset, 
                                                                    validation_dataset=full_validation_dataset, 
                                                                    test_dataset=full_test_dataset,
                                                                    is_full=True )       