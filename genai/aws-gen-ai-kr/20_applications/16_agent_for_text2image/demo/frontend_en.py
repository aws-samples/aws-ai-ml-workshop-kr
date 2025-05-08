# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to generate an image from a text prompt with the Amazon Nova Canvas model (on demand).
"""
import base64
import io
import numpy as np
from io import BytesIO
from PIL import Image
import imageio
import json
import logging
import boto3
from botocore.config import Config
import gradio as gr
from typing import Dict

import os
import shutil
import time
import pprint
from termcolor import colored
from utils import bedrock
from utils.bedrock import bedrock_info

from utils.bedrock import bedrock_model
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import eng.agentic_system_for_t2i_stepwise
import eng.agentic_system_for_t2i_text_editing
import eng.agentic_system_for_t2i_inpainting
import eng.agentic_system_for_t2i_outpainting

from textwrap import dedent

from botocore.exceptions import ClientError


# 전역 변수 정의

generated_img_path = "./generated_imgs"
edited_images_by_text_path = "./edited_imgs_by_text"

edited_images_by_mask_path = "./edited_imgs_by_mask"
inpainting_edited_images_by_mask_path = f"{edited_images_by_mask_path}/inpainting"
outpainting_edited_images_by_mask_path = f"{edited_images_by_mask_path}/outpainting"
masked_images_path = f"{edited_images_by_mask_path}/masking"

generated_images_list = []
edited_by_text_images_list = []
edited_by_mask_images_list = []

# generated_images_list = gr.State([])  
# edited_by_text_images_list = gr.State([])  
# edited_by_mask_images_list = gr.State([]) 


# Define LLMs
boto3_bedrock_us_east_1 = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    # region=os.environ.get("AWS_DEFAULT_REGION", None),
    region="us-east-1"
)

boto3_bedrock_us_west_2 = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    # region=os.environ.get("AWS_DEFAULT_REGION", None),
    region="us-west-2"
)

llm = bedrock_model(
    #model_id=bedrock_info.get_model_id(model_name="Claude-V3-5-V-2-Sonnet-CRI"),
    model_id=bedrock_info.get_model_id(model_name="Claude-V3-7-Sonnet-CRI"),
    
    bedrock_client=boto3_bedrock_us_east_1,
    stream=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    inference_config={
        'maxTokens': 4096,
        'stopSequences': ["\n\nHuman"],
        'temperature': 0.01,
    }
)

t2i_stepwise_model = bedrock_model(
    model_id=bedrock_info.get_model_id(model_name="SD-Ultra"),
    bedrock_client=boto3_bedrock_us_west_2
)

t2i_text_editing_model = bedrock_model(
    model_id=bedrock_info.get_model_id(model_name="Nova-Canvas"),
    bedrock_client=boto3_bedrock_us_east_1
)

class ImageError(Exception):
    "Custom exception for errors returned by Amazon Nova Canvas"

    def __init__(self, message):
        self.message = message


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Functions called by buttons
def convert_image_to_base64(image):
    """
    Convert various image formats to base64 string.
    
    Args:
        image: Can be PIL Image, numpy array, or dictionary containing image data
        
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        buffer = io.BytesIO()
        
        # Handle different input types
        if isinstance(image, dict):
            # If image is a dictionary, it might contain the image data
            if 'image' in image:
                image = image['image']
            else:
                raise ValueError("Dictionary does not contain image data")
                
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image_pil = Image.fromarray(image)
            image_pil.save(buffer, format="PNG")
        elif isinstance(image, Image.Image):
            # Handle PIL Image directly
            image.save(buffer, format="PNG")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
            
        # Convert to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_string
        
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        raise

def generate_mask(img):
    # imageio.imwrite("output_image.png", img["composite"])

    alpha_channel = img["layers"][0][:, :, 3]
    mask = np.where(alpha_channel == 0, 225, 0).astype(np.uint8)

    return img["background"], mask

def generate_stepwise_image(prompt, results, progress=gr.Progress()):

    analyzer_for_stepwise = eng.agentic_system_for_t2i_stepwise.genai_analyzer( 
        llm=llm,
        image_generation_model=t2i_stepwise_model
    )

    print(progress_state_at_step_1)
    progress_state_at_step_1.visible = True

    for result in progress.tqdm(analyzer_for_stepwise.invoke(ask=dedent(prompt), image_model="stable-diffusion", file_path_name=generated_img_path)):
        results.append(result)
        yield results, "Complete"

def generate_text_editing_image(original_image, edit_count, prompt, results, progress=gr.Progress()):

    original_image_path = f"{edited_images_by_text_path}/ORIGINAL_IMAGE_{edit_count}.png"

    if isinstance(original_image, np.ndarray):
        image_pil = Image.fromarray(original_image)
        image_pil.save(original_image_path, format="PNG")
    elif isinstance(original_image, Image.Image):
        original_image.save(original_image_path, format="PNG")
    else:
        raise TypeError(f"Unsupported image type: {type(original_image)}")

    print(f"Image has been successfully saved in {original_image_path}")


    analyzer_for_text_editing = eng.agentic_system_for_t2i_text_editing.GenAITextEditor(
        llm=llm,
        image_generation_model=t2i_text_editing_model
    )

    for result in progress.tqdm(
        analyzer_for_text_editing.invoke(
            ask=dedent(prompt), 
            image_model="nova-canvas", 
            file_name=f"{edited_images_by_text_path}/EDITED_IMAGE_{edit_count}.png", 
            original_image=original_image_path
        )
    ):
        results.append(result)
        yield results, "Complete"

def generate_mask_editing_image(original_image, 
                                edit_count, 
                                task_type, 
                                prompt, 
                                results, 
                                progress=gr.Progress()
    ):
    
    # 이미지 마스크 생성 및 저장
    background_image_numpy, masked_image_numpy = generate_mask(original_image)

    ## Numpy Array를 Base64 String으로 변환
    background_image_base64 = convert_image_to_base64(background_image_numpy)
    masked_image_base64 = convert_image_to_base64(masked_image_numpy)

    ## Base64 디코딩 후 BytesIO로 변환
    background_image_binary = base64.b64decode(background_image_base64)
    background_image = Image.open(BytesIO(background_image_binary))

    original_image_path = f"{edited_images_by_mask_path}/ORIGINAL_IMAGE_{edit_count}.png"
    background_image.save(original_image_path, format="PNG")
    print(f"Original Image has been successfully saved in {original_image_path}")

    masked_image_binary = base64.b64decode(masked_image_base64)
    masked_image = Image.open(BytesIO(masked_image_binary))
    
    masked_image_path = f"{masked_images_path}/MASKED_IMAGE_{edit_count}.png"
    masked_image.save(masked_image_path, format="PNG")
    print(f"Mask Image has been successfully saved in {masked_image_path}")


    if task_type == "INPAINTING":
        GenAIClass = eng.agentic_system_for_t2i_inpainting.GenAIInPainting
        file_name = f"{inpainting_edited_images_by_mask_path}/EDITED_IMAGE_{edit_count}.png"
    
    else:
        GenAIClass = eng.agentic_system_for_t2i_outpainting.GenAIOutPainting
        file_name = f"{outpainting_edited_images_by_mask_path}/EDITED_IMAGE_{edit_count}.png"

    analyzer = GenAIClass(llm=llm, image_generation_model=t2i_text_editing_model)

    common_params = {
        'ask': dedent(prompt),
        'image_model': "nova-canvas",
        'file_name': file_name,
        'mask_image': masked_image_path,
        'task_type': task_type,
        'original_image': original_image_path
    }

    # Process the results
    for result in progress.tqdm(analyzer.invoke(**common_params)):
        results.append(result)
        yield results, "Complete"

def show_progress():
    return gr.update(visible=True)

def hide_progress():
    return gr.update(visible=False)

def cleanup_state(state):
    return []

def sync_stepwise_gallery():
    return generated_images_list

def sync_text_edit_gallery():
    return edited_by_text_images_list

def sync_mask_edit_gallery():
    return edited_by_mask_images_list


def format_stepwisetaskdecomposer_output_to_html(data: Dict) -> str:
    """LangGraph의 노드 출력 데이터를 HTML로 포맷팅하여 반환"""
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .step { border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 10px; }
            .step h2 { margin: 0; font-size: 20px; color: #333; }
            .prompt { margin-top: 10px; padding: 10px; background: #f9f9f9; border-left: 4px solid #007bff; }
            .negative { color: red; font-style: }
            .meta { font-size: 14px; color: #555; }
            .description { background-color: yellow; padding: 2px 5px; }
            .positive { background: #e6ffe6; padding: 10px; border-left: 4px solid #28a745; margin-top: 10px; }
            .negative { background: #ffe6e6; padding: 10px; border-left: 4px solid #dc3545; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Image Generation Plan</h1>
    """

    for step in data.get("steps", []):
        html_content += f"""
        <div class="step">
            <h2>Step {step.get("step_number")}</h2>
            <p class="meta"><strong>Control Mode:</strong> {step.get("control_mode", "N/A")}</p>
            <p class="meta"><strong>Control Strength:</strong> {step.get("control_strength", "N/A")}</p>
            <p class="meta description"><strong>Description:</strong> {step.get("description")}</p>
            <h3>Positive Prompt</h3>
            <div class="positive">
                <p>{step["prompt"]["positive"]}</p>
            </div>
            <h3>Negative Prompt</h3>
            <div class="negative">
                <p>{step["prompt"]["negative"]}</p>
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """
    return html_content

def format_reflection_output_to_html(data: Dict) -> str:
    """Reflection 노드의 출력을 HTML로 포맷팅하여 반환"""
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .section { border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 10px; }
            .section h2 { margin: 0; font-size: 18px; color: #333; }
            .meta { font-size: 14px; color: #555; }
            .retry { font-size: 14px; color: #d9534f; font-weight: bold; }
            .retouch { background-color: yellow; padding: 2px 5px; }
            .suggestion-list { background: #f9f9f9; padding: 10px; border-left: 4px solid #007bff; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Image Reflection Report</h1>
    """
    current_step = data.get("current_step", "N/A") if data.get("should_regeneration") == "retouch" else data.get("current_step", "N/A") - 1
    

    html_content += f"""
            <div class="section">
                <h2>Step {current_step} Reflection</h2>
    """
    
    if data.get("retouch") == "true" and data.get("should_regeneration") == "pass":
        html_content += """
                <p class="retry"><strong>Retry Count:</strong> Retouch needed, but max regeneration count(2) per steps exceeded—forcing next step.</p>
        """   
    else:
        html_content += f"""
                <p class="meta"><strong>Retry Count:</strong> {data.get("retry_count", "N/A")}</p>
        """

    html_content += f"""
                <p class="retouch"><strong>Regeneration Decision:</strong> {data.get("should_regeneration", "N/A")}</p>
                <h3>Evaluation & Feedback:</h3>
                <div class="suggestion-list">
    """

    # ✅ "suggestions" 항목이 없으면 "None" 출력
    suggestions = data.get("suggestions", [])
    if suggestions:
        for idx, suggestion in enumerate(suggestions):
            html_content += f"<p><strong>Issue {idx + 1}:</strong> {suggestion}</p>"
    else:
        html_content += "<p>None</p>"

    html_content += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# Step 1 LangGraph Node
def format_prompt_reformulation_output_to_html(data: Dict) -> str:
    """PromptReformulation 노드의 출력을 HTML로 포맷팅하여 반환"""
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .section { border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 10px; }
            .section h2 { margin: 0; font-size: 18px; color: #333; }
            .control { font-size: 14px; color: #d9534f; font-weight: bold; }
            .positive { background: #e6ffe6; padding: 10px; border-left: 4px solid #28a745; margin-top: 10px; }
            .negative { background: #ffe6e6; padding: 10px; border-left: 4px solid #dc3545; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
    """
    
    html_content += f"""
            <div class="section">
                <h2>Step {data.get("current_step", "N/A")} Prompt Reformulation</h2>
                <p class="control"><strong>Control Strength:</strong> {data.get("prompt_repo", {}).get("control_strength", "N/A")}</p>
                <h3>Positive Prompt</h3>
                <div class="positive">
                    <p>{data.get("prompt_repo", {}).get("positive", "No positive prompt available.")}</p>
                </div>
                <h3>Negative Prompt</h3>
                <div class="negative">
                    <p>{data.get("prompt_repo", {}).get("negative", "No negative prompt available.")}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def format_error_message_to_html(error_message):
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .section { border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 10px; }
            .section h2 { margin: 0; font-size: 18px; color: #333; }
            .negative { background: #ffe6e6; padding: 10px; border-left: 4px solid #dc3545; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
    """
    
    html_content += f"""
            <div class="section">
                <h3>Error Message</h3>
                <div class="negative">
                    <p><strong>{error_message}</strong></p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# Step 2, 3 LangGraph Node
def format_ask_reformulation_output_to_html(data: Dict) -> str:
    """Ask Repository 노드의 출력을 HTML로 포맷팅하여 반환"""
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            .section {{ border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 10px; }}
            .section h3 {{ margin: 0; font-size: 18px; color: #333; }}
            .meta {{ font-size: 14px; color: #555; }}
            .content {{ background: #f9f9f9; padding: 10px; border-left: 4px solid #007bff; margin-top: 10px; }}
            .highlight {{ background: #fff3cd; border-left: 4px solid #ffecb5; padding: 10px; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prompt Reformulation</h1>
    """
    html_content +=f"""
            <div class="section">
                <h3>Original Prompt</h3>
                <div class="content">
                    <!--<p><strong>Origin Prompt:</strong> {data.get("origin_ask_repo", "No origin prompt available.")}</p>-->
                    <p>{data.get("origin_ask_repo", "No origin prompt available.")}</p>
                </div>
                <h3>Reformed Prompt</h3>
                <div class="content">
                    <!--<p><strong>Reformed Prompt:</strong> {data.get("ask_repo", "No reformed prompt available.")}</p>-->
                    <p>{data.get("ask_repo", "No reformed prompt available.")}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
    
def format_check_readness_prompt_generation_output_to_html(data: Dict) -> str:
    """Prompt Components 노드의 출력을 HTML로 포맷팅하여 반환"""
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .section { border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 10px; }
            .section h3 { margin: 0; font-size: 18px; color: #333; }
            .content { background: #f9f9f9; padding: 10px; border-left: 4px solid #007bff; margin-top: 10px; }
            .highlight { background: #fff3cd; border-left: 4px solid #ffecb5; padding: 10px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Visual Component Extraction</h1>
    """
    html_content +=f"""
            <div class="section">
                <h3>Subject</h3>
                <div class="content">
                    <p>{data.get("prompt_components", {}).get("subject", {}).get("content", "None")}</p>
                </div>
                <h3>Action</h3>
                <div class="content">
                    <p>{data.get("prompt_components", {}).get("action", {}).get("content", "None")}</p>
                </div>
                <h3>Environment</h3>
                <div class="content">
                    <p>{data.get("prompt_components", {}).get("environment", {}).get("content", "None")}</p>
                </div>
                <h3>Lighting</h3>
                <div class="content">
                    <p>{data.get("prompt_components", {}).get("lighting", {}).get("content", "None")}</p>
                </div>
                <h3>Style</h3>
                <div class="content">
                    <p>{data.get("prompt_components", {}).get("style", {}).get("content", "None")}</p>
                </div>
                <h3>Camera Position</h3>
                <div class="content">
                    <p>{data.get("prompt_components", {}).get("camera_position", {}).get("content", "None")}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def format_prompt_generation_for_image_output_to_html(data: Dict) -> str:
    """Image Prompt 노드의 출력을 HTML로 포맷팅하여 반환"""
    html_content = """
    <html>
    <head>
         <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .section { border: 2px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 10px; }
            .section h3 { margin: 0; font-size: 18px; color: #333; }
            .content { background: #f9f9f9; padding: 10px; border-left: 4px solid #007bff; margin-top: 10px; }
            .negative { background: #ffe6e6; padding: 10px; border-left: 4px solid #dc3545; margin-top: 10px; }
            .highlight { background: #fff3cd; padding: 10px; border-left: 4px solid #ffecb5; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Optimized Image Editing Prompt Generation</h1>
    """
    html_content +=f"""
            <div class="section">
                <h3>Main Prompt</h3>
                <div class="content">
                    <p>{data.get("image_prompt", {}).get("main_prompt", "No main prompt available.")}</p>
                </div>
                <h3>Negative Prompt</h3>
                <div class="negative">
                    <p>{data.get("image_prompt", {}).get("negative_prompt", "No negative prompt available.")}</p>
                </div>
                <h3>Mask Prompt</h3>
                <div class="highlight">
                    <p>{data.get("image_prompt", {}).get("mask_prompt", "Since the image mask was given manually, the model did not generate a mask prompt.")}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

html_content_for_lightmode = """
    <div class="lightmode-notice">
        <span class="icon">⚠️</span>
        <span class="text">
            For the best demo experience, please set your browser to <strong>Light Mode</strong>.
        </span>
    </div>

    <style>
    .lightmode-notice {
        display: flex;
        align-items: center;
        background-color: #fffbea;
        color: #4b4b4b;
        font-size: 14px;
        padding: 12px 16px;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }

    .lightmode-notice .icon {
        margin-right: 10px;
        font-size: 18px;
    }

    .lightmode-notice .text {
        flex: 1;
        line-height: 1.5;
    }
    </style>
"""


# UI
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Text to Image Generation with AI Agent</h1>")


    with gr.Tab("1️⃣ Generate Image"):
        results_at_step_1 = gr.State([])

        prompt = gr.Textbox(
                placeholder="Please describe your creative ideas for the image", 
                label="Prompt"
            )
        
        with gr.Row():
            generate_image_btn = gr.Button("Generate")
            clear_btn_at_step_1 = gr.ClearButton([prompt])
        
        html_browser_light_mode = gr.HTML(html_content_for_lightmode)

        @gr.render(inputs=results_at_step_1)
        def render_results(graph_results):
            
            for result in graph_results:
                key = result["key"]
                value = result["value"]
                if key == "StepwiseTaskDecomposer":
                    if value["prev_node"] != "Reflection":
                        html_stepwise_value = format_stepwisetaskdecomposer_output_to_html(value)
                        stepwise = gr.HTML(html_stepwise_value)
                    else:
                        pass
                
                elif key == "ImageGeneration":
                    image_generation_step = gr.HTML(f"<h1>Step {value['current_step']} Image Generation<h1>")
                    image_block = gr.Image(value['condition_image'], type="filepath", label="Generated Image")
                    generated_images_list.append(value['condition_image'])

                elif key == "Reflection":
                    html_reflection_value = format_reflection_output_to_html(value)
                    reflection = gr.HTML(html_reflection_value)

                elif key == "PromptReformulation":
                    prompt_reformulation_label = gr.HTML("<h1>Prompt Reformulation Based on Reflection Report</h1>")
                    if value['error'] == True:
                        html_error_message_value_at_step_1 = format_error_message_to_html(value['error_message'])
                        error_message_block_at_step_1 = gr.HTML(html_error_message_value_at_step_1)
                    else:
                        html_reformulation_value = format_prompt_reformulation_output_to_html(value)
                        reformulation = gr.HTML(html_reformulation_value)

                else:
                    json_block = gr.JSON(value, label=f"Output from node '{key}'")

        progress_state_at_step_1 = gr.Textbox(label="Progress State", container=False, visible=False)

        generate_image_btn.click(
            fn=show_progress, 
            inputs=[], 
            outputs=[progress_state_at_step_1]
        )

        generate_image_btn.click(
            fn=generate_stepwise_image, 
            inputs=[prompt, results_at_step_1],
            outputs=[results_at_step_1, progress_state_at_step_1],
            show_progress="full",
            show_progress_on=progress_state_at_step_1
        )

        clear_btn_at_step_1.click(
            fn=hide_progress,
            inputs=[],
            outputs=[progress_state_at_step_1]
        )

        clear_btn_at_step_1.click(
            fn=cleanup_state,
            inputs=results_at_step_1,
            outputs=results_at_step_1
        )

    
    with gr.Tab("2️⃣ Edit Image By Text"):
        results_at_step_2 = gr.State([])
        edit_count_at_step_2 = gr.State(0)

        with gr.Row(): # Inputs
            original_image = gr.Image(label="Base Image")
            prompt = gr.Textbox(placeholder="Please describe how to edit the generated image", label="Prompt")
        
        with gr.Row():
            edit_image_btn = gr.Button("Edit")
            clear_btn_at_step_2 = gr.ClearButton([original_image, prompt])

        html_browser_light_mode = gr.HTML(html_content_for_lightmode)

        @gr.render(inputs=results_at_step_2)
        def render_results(graph_results):
            if isinstance(graph_results, gr.State):
                graph_results = graph_results.value  # Extract the value from gr.State
            for result in graph_results:
                key = result["key"]
                value = result["value"]
                if key == "ask_reformulation":
                    pass

                elif key == "check_readness_prompt_generation":
                    html_check_readiness_prompt_generation = format_check_readness_prompt_generation_output_to_html(value)
                    check_readiness_prompt_generation = gr.HTML(html_check_readiness_prompt_generation)

                elif key == "prompt_generation_for_image":
                    html_prompt_generation_for_image = format_prompt_generation_for_image_output_to_html(value)
                    prompt_generation_for_image = gr.HTML(html_prompt_generation_for_image)

                elif key == 'image_generation':
                    image_generation_label_at_step2 = gr.HTML(f"<h1>Image Generation<h1>")
                    if value["error"] == True:
                        html_error_message_value_at_step_2 = format_error_message_to_html(value['error_message'])
                        error_message_block_at_step_2 = gr.HTML(html_error_message_value_at_step_2)
                    else:
                        image_block = gr.Image(value['generated_img_path'], type="filepath")
                        edited_by_text_images_list.append(value['generated_img_path'])

                else:
                    json_block = gr.JSON(value, label=f"Output from node '{key}'")
      
        progress_state_at_step_2 = gr.Textbox(label="Progress State", container=False, visible=False)

        edit_image_btn.click(
            fn=show_progress, 
            inputs=[], 
            outputs=[progress_state_at_step_2]
        )

        edit_image_btn.click(
            fn=generate_text_editing_image, 
            inputs=[original_image, edit_count_at_step_2, prompt, results_at_step_2], 
            outputs=[results_at_step_2, progress_state_at_step_2],
            show_progress="full",
            show_progress_on=progress_state_at_step_2
        )

        edit_image_btn.click(
            fn=lambda x: x + 1,
            inputs=[edit_count_at_step_2],
            outputs=[edit_count_at_step_2]
        )

        clear_btn_at_step_2.click(
            fn=cleanup_state,
            inputs=results_at_step_2,
            outputs=results_at_step_2
        )

        clear_btn_at_step_2.click(
            fn=hide_progress,
            inputs=[],
            outputs=[progress_state_at_step_2]
        )
        

    with gr.Tab("3️⃣ Edit Image By Mask"):
        results_at_step_3 = gr.State([])
        edit_count_at_step3 = gr.State(0)

        with gr.Row(): # Inputs
            original_image = gr.ImageMask(sources=["upload"], layers=False, transforms=[], format="png", label="Base Image", show_label=True)
            with gr.Column():
                task_type = gr.Radio(["INPAINTING", "OUTPAINTING"], value="INPAINTING", label="Task Type")
                prompt = gr.Textbox(placeholder="Please describe how to edit the generated image", label="Prompt")
            
        with gr.Row(): # Buttons
            edit_image_by_mask_btn = gr.Button("Edit")
            clear_btn_at_step_3 = gr.ClearButton([original_image, task_type, prompt])

        html_browser_light_mode = gr.HTML(html_content_for_lightmode)
        
        @gr.render(inputs=results_at_step_3)
        def render_results(graph_results):
            for result in graph_results:
                key = result["key"]
                value = result["value"]
                if key == "ask_reformulation":
                    pass

                elif key == "check_readness_prompt_generation":
                    html_check_readiness_prompt_generation_at_step_3 = format_check_readness_prompt_generation_output_to_html(value)
                    check_readiness_prompt_generation_at_step_3 = gr.HTML(html_check_readiness_prompt_generation_at_step_3)

                elif key == 'image_generation':
                    image_generation_label_at_step_3 = gr.HTML(f"<h1>Image Generation<h1>")
                    if value["error"] == True:
                        html_error_message_value_at_step_3 = format_error_message_to_html(value['error_message'])
                        error_message_block_at_step_3 = gr.HTML(html_error_message_value_at_step_3)
                    else:
                        image_block_at_step_3 = gr.Image(value['generated_img_path'], type="filepath")
                        edited_by_mask_images_list.append(value['generated_img_path'])
                    
                elif key == "prompt_generation_for_image":
                    html_prompt_generation_for_image_at_step_3 = format_prompt_generation_for_image_output_to_html(value)
                    prompt_generation_for_image_at_step_3 = gr.HTML(html_prompt_generation_for_image_at_step_3)

                else:
                    json_block = gr.JSON(value, label=f"Output from node '{key}'")
        
        progress_state_at_step_3 = gr.Textbox(label="Progress State", container=False, visible=False)

        edit_image_by_mask_btn.click(
            fn=show_progress, 
            inputs=[], 
            outputs=[progress_state_at_step_3]
        )

        edit_image_by_mask_btn.click(
            fn=generate_mask_editing_image, 
            inputs=[original_image, edit_count_at_step3, task_type, prompt, results_at_step_3], 
            outputs=[results_at_step_3, progress_state_at_step_3],
            show_progress="full",
            show_progress_on=progress_state_at_step_3
        )

        edit_image_by_mask_btn.click(
            fn=lambda x: x + 1,
            inputs=[edit_count_at_step3],
            outputs=[edit_count_at_step3]
        )

        clear_btn_at_step_3.click(
            fn=cleanup_state,
            inputs=results_at_step_3,
            outputs=results_at_step_3
        )

        clear_btn_at_step_3.click(
            fn=hide_progress,
            inputs=[],
            outputs=[progress_state_at_step_3]
        )
        


def recreate_directory(directory_path):
    # Remove directory and all its contents if it exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    
    # Create new directory
    os.makedirs(directory_path)


# start interface
if __name__ == "__main__":
    recreate_directory(generated_img_path)
    recreate_directory(edited_images_by_text_path)

    recreate_directory(edited_images_by_mask_path)
    recreate_directory(inpainting_edited_images_by_mask_path)
    recreate_directory(outpainting_edited_images_by_mask_path)
    recreate_directory(masked_images_path)

    demo.launch(share=False)
    