

def format_text(text):
    """마침표를 기준으로 줄바꿈을 추가하여 텍스트를 포맷팅하는 함수입니다."""
    # 기존 줄바꿈 문자를 공백으로 대체
    text = text.replace('\n', ' ')
    
    # 연속된 공백을 하나의 공백으로 정리
    text = ' '.join(text.split())
    
    # 마침표와 공백이 있는 패턴을 찾아서 줄바꿈으로 대체
    formatted_text = text.replace('. ', '.\n')
    
    # 불필요한 빈 줄 제거
    formatted_text = '\n'.join(line.strip() for line in formatted_text.splitlines() if line.strip())
    
    print(formatted_text)


def pretty_print_df(df):
    # sample_ground_truth = df.head(1).ground_truth.iloc[0]
    # sample_ground_truth = df.ground_truth.iloc[0]
    question = df.question.iloc[0]
    print("question:")
    # print_formatted_text(question)
    format_text(question)    

    sample_ground_truth = df.ground_truth.iloc[0]
    print("ground_truth:")
    format_text(sample_ground_truth)

    question_type = df.question_type.iloc[0]
    print("question_type:")
    format_text(question_type)

    contexts = df.contexts.iloc[0]
    print("contexts:")
    format_text(contexts)

import re
import ast

def clean_string(s):
    # s = re.sub(r'[^\x00-\x7F]+', '', s) <-- 이는 유니코드 특수문자, 이모지, 한글 등 ASCII가 아닌 모든 문자를 제거
    s = s.replace("'", '"') # 이는 문자열을 JSON 형식과 호환되게 만들거나, ast.literal_eval()이 문자열을 더 잘 처리할 수 있게 하기 위함입니다
    return s    

def convert_to_list(example):
    cleaned_context = clean_string(example["contexts"])
    # cleaned_context = example["contexts"]
    # print("cleaned_context: ", cleaned_context)
    try:
        contexts = ast.literal_eval(cleaned_context)
    except (ValueError, SyntaxError) as e:
        contexts = cleaned_context
        # print(f"Error: {str(e)}")
    return {"contexts": contexts}


def print_formatted_text(text):
    # text를 줄바꿈 문자를 기준으로 분리하여 각 줄을 출력
    lines = text.split('\n')
    
    for line in lines:
        # 빈 줄은 그대로 출력하고, 내용이 있는 줄은 양쪽 공백을 제거하여 출력
        if line.strip():
            print(line.strip())
        else:
            print()

def generate_answer(question, contexts, boto3_client, model_id):
    system_prompt = """You are an AI assistant that uses retrieved context to answer questions accurately. 
    Follow these guidelines:
    1. Use the provided context to inform your answers.
    2. If the context doesn't contain relevant information, say "I don't have enough information to answer that."
    3. Be concise and to the point in your responses."""

    user_prompt = f"""Context: {contexts}

    Question: {question}

    Please answer the question based on the given context."""

    response = boto3_client.converse(
        modelId=model_id,
        messages=[{'role': 'user', 'content': [{'text': user_prompt}]}],
        system=[{'text': system_prompt}]
    )

    answer = response['output']['message']['content'][0]['text']
    return answer

import json
import pandas as pd
from datasets import Dataset

def merge_dataset_and_results(dataset: Dataset, results_path: str) -> pd.DataFrame:
    # Dataset을 pandas DataFrame으로 변환
    df_dataset = dataset.to_pandas()
    
    # 결과 JSON 읽기
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # detailed_results를 DataFrame으로 변환
    df_results = pd.DataFrame(results['detailed_results'])
    
    # row 번호를 0부터 시작하도록 조정 (DataFrame의 인덱스와 맞추기 위해)
    df_results['row'] = df_results['row'] - 1
    df_results = df_results.set_index('row')
    
    # Dataset DataFrame과 결과 DataFrame 병합
    # 데이터셋의 인덱스와 결과의 row 값을 기준으로 병합
    merged_df = pd.concat([df_dataset, df_results], axis=1)
    
    # # average_scores를 별도의 컬럼으로 추가 (전체 평균 점수)
    # for metric, score in results['average_scores'].items():
    #     merged_df[f'avg_{metric}'] = score
    
    return merged_df


def show_sample_row_eval(df, row):
    user_input = df.iloc[row].user_input
    reference = df.iloc[row].reference
    response = df.iloc[row].response
    answer_relevancy = df.iloc[row].answer_relevancy
    faithfulness = df.iloc[row].faithfulness
    context_recall = df.iloc[row].context_recall
    context_precision = df.iloc[row].context_precision

    print("## Question: \n", user_input)
    print("## Ground_truth: \n", reference)
    print("## Generated Answer: \n", response)
    print("## AnswerRelevancy: ", answer_relevancy)
    print("## Faithfulness: ", faithfulness)
    print("## ContextRecall: ", context_recall)
    print("## ContextPrecision: ", context_precision)

    all_response = list()
    all_response.append("## Question: " + user_input)
    all_response.append("## Ground_truth: " + reference)
    all_response.append("## Generated Answer: " + response)
    all_response.append("## AnswerRelevancy: " + str(answer_relevancy))    
    all_response.append("## Faithfulness: " + str(faithfulness))        
    all_response.append("## ContextRecall: " + str(context_recall))            
    all_response.append("## ContextPrecision: " + str(context_precision))   

    return all_response              




    # print("## Question: \n", df.iloc[row].user_input)
    # print("## Ground_truth: \n", df.iloc[row].reference)
    # print("## Generated Answer: \n", df.iloc[row].response)
    # print("## AnswerRelevancy: ", df.iloc[row].answer_relevancy)
    # print("## Faithfulness: ", df.iloc[row].faithfulness)
    # print("## ContextRecall: ", df.iloc[row].context_recall)
    # print("## ContextPrecision: ", df.iloc[row].context_precision)
