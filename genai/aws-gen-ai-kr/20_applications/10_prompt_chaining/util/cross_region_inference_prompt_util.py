from langchain import PromptTemplate
import os
import re
import json
from jinja2 import Template
from string import Template
from tqdm import tqdm
import boto3
import time

# claude 3.5 model inference
# model_id = anthropic.claude-3-5-sonnet-20240620-v1:0

# Bedrock cross-region inference
primary_region = "us-east-1"
# primary_region = "us-west-2"
inferenceProfileId = 'us.anthropic.claude-3-5-sonnet-20240620-v1:0'

# parameters to consolidate inpute/output tokens after chaining prompt invoke calls
merged_total_input_tokens = 0
merged_total_output_tokens = 0

def print_json(data):
    print(json.dumps(data, indent = 3,ensure_ascii=False))

def load_template_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        template_content = file.read()
    return template_content



def invoke_claude(prompt):
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=primary_region)
    body = json.dumps({
        "max_tokens": 4196,
        "messages": [{"role": "user", "content": prompt}],
        "anthropic_version": "bedrock-2023-05-31"
    })

    # time.sleep(60)  # Add a small delay to avoid rate limiting
    response = bedrock.invoke_model(body=body, modelId=inferenceProfileId)

    response_body = json.loads(response.get("body").read())
    output_text = response_body.get("content")[0].get('text')

    # print(output_text)
    # print("호출")
    
    input_token_count = int(response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-input-token-count"])
    output_token_count = int(response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-output-token-count"])
    
    return output_text, input_token_count, output_token_count

def print_slide_titles(json_data):
    # JSON 문자열을 Python 딕셔너리로 파싱
    data = json.loads(json_data) if isinstance(json_data, str) else json_data
    
    # 메인 타이틀 출력
    print(f"프레젠테이션 제목: {data['title']}\n")
    
    # 슬라이드 리스트를 순회하며 각 슬라이드의 제목 출력
    for slide in data['slides']:
        print(f"슬라이드 {slide['slide_number']}: {slide['slide_title']}")


def create_slide_prompt(slide_prompt_template, topic, outline, start_slide, end_slide):
    slide_prompt = slide_prompt_template.render(
        topic = topic,
        outline = outline,
        start_slide = start_slide,
        end_slide = end_slide
    )
    return slide_prompt


def generate_slide_content(slide_prompt, outline, output_dir, include_outline, start_slide, end_slide, tries):

    slides = [] 
    total_input_tokens = 0 
    total_output_tokens =0
    i=tries
    
    slide_content, input_tokens, output_tokens = invoke_claude(slide_prompt)

    slides.append(f"{slide_content}")

    # Print all slides
    print("\n Detailed Slide Contents:")
    for slide in slides:
        print("\n" + "-"*50 + "\n")
        print(slide)
        print("\n" + "-"*50 + "\n")

    print(f"## Input tokens: {input_tokens}, Output tokens: {output_tokens} \n")
        
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    # Update global counters
    global merged_total_input_tokens
    global merged_total_output_tokens

    merged_total_input_tokens += total_input_tokens
    merged_total_output_tokens += total_output_tokens

    # time.sleep(120)  # Add a small delay to avoid rate limiting
        
    print("\n" + "="*50 + "\n")
    print("PowerPoint presentation generation complete!")
    print(f"Input Tokens: {total_input_tokens}")
    print(f"Output Tokens: {total_output_tokens}")

    if include_outline: 
        output_file = f"result_include_outline_each_slide_{i}.txt"
    else:
        output_file = f"result_exclude_outline_each_slide_{i}.txt"
    
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        for slide in slides:
            f.write(slide)
            f.write("\n" + "-"*50 + "\n")
            f.write(f"\n\nInput Tokens: {total_input_tokens}\n")
            f.write(f"\n\nOutput Tokens: {total_output_tokens}\n")

    print(f"Processed and saved: {output_path}")


def merge_files(output_directory):
    output_file = 'final_result.txt'  # Default output file name
    try:
        # Ensure the directory exists
        if not os.path.exists(output_directory):
            print(f"The directory '{output_directory}' does not exist.")
            return

        # Get list of files in the directory
        files = os.listdir(output_directory)

        # Open the output file in write mode
        with open(os.path.join(output_directory, output_file), 'w', encoding='utf-8') as outfile:
            # Iterate through each file in the directory
            for filename in files:
                filepath = os.path.join(output_directory, filename)
                # Check if it's a file (not a subdirectory) and not the output file itself
                if os.path.isfile(filepath) and filename != output_file:
                    # Write the filename as a header
                    outfile.write(f"\n\n--- Contents of {filename} ---\n\n")
                    # Open each file and append its contents to the output file
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
            outfile.write("\n" + "="*50 + "\n")
            outfile.write(f"\n\nTotal Input Tokens: {merged_total_input_tokens}\n")
            outfile.write(f"\n\nTotal Output Tokens: {merged_total_output_tokens}\n")            

        print(f"All files have been merged into '{output_file}' in the directory '{output_directory}'.")
        

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def invoke_all_claude(prompt):
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=primary_region)
    body = json.dumps({
        "max_tokens": 4196,
        "messages": [{"role": "user", "content": prompt}],
        "anthropic_version": "bedrock-2023-05-31"
    })
    
    response = bedrock.invoke_model(body=body, modelId=inferenceProfileId)
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get('text'), int(response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-input-token-count"]), int(response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-output-token-count"])


def generate_all_slide_content(presentation_prompt, output_dir):

    presentation_content, total_input_tokens, total_output_tokens = invoke_all_claude(presentation_prompt)

    print("Generated Presentation Content:")
    print(presentation_content)
    print("\n" + "="*50 + "\n")
    print(f"Total output tokens: {total_output_tokens}")
    print("PowerPoint presentation generation complete!")

    # Optional: Split the content into individual slides
    slides = presentation_content.split("Slide")[1:]  # Split by "Slide" and remove empty first element
    slides = ["Slide" + slide for slide in slides]  # Add "Slide" back to each element

    print("\nIndividual Slides:")
    # for slide in slides:
    #     print(slide.strip())
    #     print("\n" + "-"*50 + "\n")


    output_file = f"result_all_slide.txt"
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(presentation_content)
        f.write(f"\n\nTotal Input Tokens: {total_input_tokens}\n")
        f.write(f"\n\nTotal Output Tokens: {total_output_tokens}\n")

    print(f"Processed and saved: {output_path}")