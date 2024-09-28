import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

def load_model_and_tokenizer(model_path):
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_text(pipeline, prompt, max_length=100):
    '''
    model, tokenizer = load_model_and_tokenizer(model_path)

    # 텍스트 생성 파이프라인 생성
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    # 추론 예시
    prompt = "다음은 뉴스 기사의 요약입니다:"
    generated_text = generate_text(pipeline, prompt)
    print(f"Generated text:\n{generated_text}")
    '''
    generated_text = pipeline(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )[0]['generated_text']
    
    return generated_text

def generate_response(messages, model, tokenizer, full_test_dataset, max_new_tokens, article_num):
    input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id= tokenizer.eos_token_id,
        # do_sample=True,
        temperature=0.1,
        # top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    generated_answer = tokenizer.decode(response,skip_special_tokens=True)

    original_answer = full_test_dataset[article_num]['messages'][2]['content']
    
    print(f"**Query:**\n{full_test_dataset[article_num]['messages'][1]['content']}\n")
    # print(f"**Query:**\n{test_dataset[rand_idx]['text'][1]['content']}\n")
    # print(f"**Original Answer:**\n{test_dataset[rand_idx]['text'][2]['content']}\n")
    print(f"**Original Answer:**\n{original_answer}\n")
    print(f"**Generated Answer:**\n{generated_answer}")
    
    return generated_answer, original_answer

from datasets import load_dataset 
from random import randint

def get_message_from_dataset(sample_dataset_json_file, article_num=75):
    # Load our test dataset
    full_test_dataset = load_dataset("json", data_files=sample_dataset_json_file, split="train")
    print("lenght of full test dataset: ", len(full_test_dataset))

    # Test on sample 
    # rand_idx = randint(0, len(full_test_dataset)-1)
    # rand_idx = 75
    # print("rand_idx: ", rand_idx)
    messages = full_test_dataset[article_num]["messages"][:2]
    # messages = test_dataset[rand_idx]["text"][:2]
    print("messages: \n", messages)

    return messages, full_test_dataset



from rouge import Rouge
from bert_score import score
import numpy as np

def single_evaluate(generated_summary, original_summary):

    quality_score = evaluate_summary(generated_summary, original_summary)
    
    return quality_score

import json
def evaluate_summary(generated_summary, original_answer):
    # ROUGE 점수 계산
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_summary, original_answer)[0]
    
    # BERT Score 계산
    P, R, F1 = score([generated_summary], [original_answer], lang="ko", verbose=False)
    
    # 결과를 딕셔너리로 구성 (소수점 3자리까지 반올림)
    result = {
        "rouge_scores": {
            "rouge-1": round(rouge_scores['rouge-1']['f'], 3),
            "rouge-2": round(rouge_scores['rouge-2']['f'], 3),
            "rouge-l": round(rouge_scores['rouge-l']['f'], 3)
        },
        "bert_score": {
            "precision": round(P.mean().item(), 3),
            "recall": round(R.mean().item(), 3),
            "f1": round(F1.mean().item(), 3)
        }
    }
    
    # 전반적인 품질 점수 계산
    overall_score = np.mean([
        rouge_scores['rouge-1']['f'],
        rouge_scores['rouge-2']['f'],
        rouge_scores['rouge-l']['f'],
        F1.mean().item()
    ])
    
    result["overall_ro-1-2-l_bert_score"] = round(overall_score, 3)
    
    # JSON 형식으로 출력
    print(json.dumps(result, indent=2))
    
    return result


import json
import random
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from bert_score import score
import numpy as np
import os
from datetime import datetime

class SummaryEvaluationSystem:
    def __init__(self, result_folder):
        self.rouge = Rouge()
        self.result_folder = self.create_result_folder(result_folder)
        os.makedirs(result_folder, exist_ok=True)

    def create_result_folder(self, result_folder):
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%Y-%m-%d-%H-%M-%S")
        output_directory = os.path.join(result_folder, currentTime)
        os.makedirs(output_directory, exist_ok=True)
        print(f"output_directory: {output_directory}")

        return output_directory

    def extract_test_samples(self, test_dataset_json, sample_test_num):
        """
        JSON 테스트 데이터셋에서 지정된 수만큼의 샘플을 추출합니다.
        
        :param test_dataset_json: JSON 형식의 테스트 데이터셋
        :param sample_test_num: 추출할 샘플 수
        :return: 추출된 메시지 리스트
        """
        # JSON에서 데이터 로드 및 샘플 추출 로직
        full_test_dataset = load_dataset("json", data_files=test_dataset_json, split="train")
        print("lenght of full test dataset: ", len(full_test_dataset))
        
        
        dataset_sample = full_test_dataset.select(range(sample_test_num))

        message_list = []
        for data in dataset_sample:
            message = data["messages"]
            message_list.append(message)
            # print("messages: ", message)
        
        return message_list


    def generate_summaries(self, message_list, model, tokenizer, verbose=False):
        """
        주어진 메시지 리스트에 대해 요약을 생성합니다.
        
        :param message_list: 원본 메시지 리스트
        :param model: 요약 생성에 사용할 모델
        :param tokenizer: 텍스트 토큰화에 사용할 토크나이저
        :return: 생성된 요약 리스트, 원본 요약 리스트
        """
        generated_answer_list = []
        original_answer_list = []
        for msg in message_list:
            # 모델을 사용한 요약 생성 로직
            input_ids = tokenizer.apply_chat_template(msg,add_generation_prompt=True,return_tensors="pt").to(model.device)
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id= tokenizer.eos_token_id,
                # do_sample=True,
                temperature=0.1,
                # top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            generated_answer = tokenizer.decode(response,skip_special_tokens=True)
            generated_answer_list.append(generated_answer)

            original_answer = msg[2]['content']
            original_answer_list.append(original_answer)
            
            query = msg[1]['content']

            if verbose:
                print(f"**Query:**\n{query}\n")
                # print(f"**Query:**\n{test_dataset[rand_idx]['text'][1]['content']}\n")
                # print(f"**Original Answer:**\n{test_dataset[rand_idx]['text'][2]['content']}\n")
                print(f"**Original Answer:**\n{original_answer}\n")
                print(f"**Generated Answer:**\n{generated_answer}")
                
                
        return generated_answer_list, original_answer_list
        
        
    def _evaluate_summary(self, generated_summary, original_answer):
        # ROUGE 점수 계산
        rouge = Rouge()
        rouge_scores = rouge.get_scores(generated_summary, original_answer)[0]

        # BERT Score 계산
        P, R, F1 = score([generated_summary], [original_answer], lang="ko", verbose=False)

        # 결과를 딕셔너리로 구성 (소수점 3자리까지 반올림)
        result = {
            "rouge_scores": {
                "rouge-1": round(rouge_scores['rouge-1']['f'], 3),
                "rouge-2": round(rouge_scores['rouge-2']['f'], 3),
                "rouge-l": round(rouge_scores['rouge-l']['f'], 3)
            },
            "bert_score": {
                "precision": round(P.mean().item(), 3),
                "recall": round(R.mean().item(), 3),
                "f1": round(F1.mean().item(), 3)
            }
        }

        # 전반적인 품질 점수 계산
        overall_score = np.mean([
            rouge_scores['rouge-1']['f'],
            rouge_scores['rouge-2']['f'],
            rouge_scores['rouge-l']['f'],
            F1.mean().item()
        ])

        result["overall_ro-1-2-l_bert_score"] = round(overall_score, 3)

        # JSON 형식으로 출력
        # print(json.dumps(result, indent=2))

        return result



    def summarize_results(self, results, model_name):
        df = pd.DataFrame(results)

        # 평균 점수 계산
        avg_scores = {
            'rouge-1': df['rouge_scores'].apply(lambda x: x['rouge-1']).mean(),
            'rouge-2': df['rouge_scores'].apply(lambda x: x['rouge-2']).mean(),
            'rouge-l': df['rouge_scores'].apply(lambda x: x['rouge-l']).mean(),
            'bert_precision': df['bert_score'].apply(lambda x: x['precision']).mean(),
            'bert_recall': df['bert_score'].apply(lambda x: x['recall']).mean(),
            'bert_f1': df['bert_score'].apply(lambda x: x['f1']).mean(),
            'overall_score': df['overall_ro-1-2-l_bert_score'].mean()
        }

        # 결과 출력
        # print("평균 평가 점수:")
        # print(json.dumps(avg_scores, indent=2))

        # CSV 파일로 저장
        df_export = pd.DataFrame({
            'summary_id': df['summary_id'],
            'rouge-1': df['rouge_scores'].apply(lambda x: x['rouge-1']),
            'rouge-2': df['rouge_scores'].apply(lambda x: x['rouge-2']),
            'rouge-l': df['rouge_scores'].apply(lambda x: x['rouge-l']),
            'bert_precision': df['bert_score'].apply(lambda x: x['precision']),
            'bert_recall': df['bert_score'].apply(lambda x: x['recall']),
            'bert_f1': df['bert_score'].apply(lambda x: x['f1']),
            'overall_score': df['overall_ro-1-2-l_bert_score']
        })
        
        # print("df_export: \n", df_export)
        result_path = os.path.join(self.result_folder, f"{model_name}_evaluation_results.csv")
        df_export.to_csv(result_path, index=False)

        self.evaluation_results = result_path
        print(f"\n평가 상세 결과가 {result_path} 파일로 저장되었습니다.")
    
    
    def evaluate_summaries(self, generated_summaries, original_summaries):
        """
        생성된 요약과 원본 요약을 비교하여 평가 점수를 계산합니다.
        
        :param generated_summaries: 생성된 요약 리스트
        :param original_summaries: 원본 요약 리스트
        :return: 평가 점수 딕셔너리 리스트
        """
        # ROUGE 및 BERT Score 계산 로직
        results = []
        for i, (generated, original) in enumerate(zip(generated_summaries,original_summaries) ):
            result = self._evaluate_summary(generated, original)
            result['summary_id'] = i
            results.append(result)
        return results
    
    def calculate_average_scores(self, results, model_name):
        """
        평가 결과의 평균 점수를 계산합니다.

        :param results: 평가 결과 딕셔너리 리스트
        :return: 평균 점수 딕셔너리 (JSON 문자열)
        """
        avg_scores = {
            "rouge_scores": {
                "rouge-1": 0,
                "rouge-2": 0,
                "rouge-l": 0
            },
            "bert_score": {
                "precision": 0,
                "recall": 0,
                "f1": 0
            },
            "overall_ro-1-2-l_bert_score": 0
        }

        n = len(results)

        for result in results:
            for key in avg_scores["rouge_scores"]:
                avg_scores["rouge_scores"][key] += result["rouge_scores"][key]
            for key in avg_scores["bert_score"]:
                avg_scores["bert_score"][key] += result["bert_score"][key]
            avg_scores["overall_ro-1-2-l_bert_score"] += result["overall_ro-1-2-l_bert_score"]

        # 평균 계산
        for key in avg_scores["rouge_scores"]:
            avg_scores["rouge_scores"][key] = round(avg_scores["rouge_scores"][key] / n, 3)
        for key in avg_scores["bert_score"]:
            avg_scores["bert_score"][key] = round(avg_scores["bert_score"][key] / n, 3)
        avg_scores["overall_ro-1-2-l_bert_score"] = round(avg_scores["overall_ro-1-2-l_bert_score"] / n, 3)

        json_string = json.dumps(avg_scores, indent=2)

        # 파일로 저장
        summary_json_file = os.path.join(self.result_folder, f"{model_name}_summary_results.json")
        with open(summary_json_file, 'w', encoding='utf-8') as f:
            f.write(json_string)
        
        print(f"JSON 데이터가 '{summary_json_file}' 파일로 저장되었습니다.")
        self.summary_json_file = summary_json_file

        # JSON 문자열로 변환
        return json_string

    def run_evaluation(self, test_dataset_json, sample_test_num, model_name, model, tokenizer):
        """
        전체 평가 프로세스를 실행합니다.
        
        :param test_dataset_json: JSON 형식의 테스트 데이터셋
        :param sample_test_num: 평가할 샘플 수
        :param model: 요약 생성에 사용할 모델
        :param tokenizer: 텍스트 토큰화에 사용할 토크나이저
        """
        message_list = self.extract_test_samples(test_dataset_json, sample_test_num)
        generated_summaries, original_summaries = self.generate_summaries(message_list, model, tokenizer)
        # print("generated_summaries: \n", generated_summaries)
        # print("original_summaries: \n", original_summaries)
        evaluation_results = self.evaluate_summaries(generated_summaries, original_summaries)
        # print("evaluation_results: \n", evaluation_results)
        self.summarize_results(evaluation_results, model_name)

        
        # 평균 점수 계산 및 출력
        avg_scores = self.calculate_average_scores(evaluation_results, model_name)
        print("평균 평가 점수:")
        print(avg_scores)

        return self.summary_json_file, self.evaluation_results
