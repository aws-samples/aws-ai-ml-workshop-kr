# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Create an image with the Amazon Titan Image Generators"""
# Python Built-Ins:
import os
from typing import Optional

# External Dependencies:
import json
import boto3
import base64
import random
from textwrap import dedent
from botocore.config import Config
from botocore.exceptions import ClientError

def generate_image(prompt):
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Set the model ID, e.g., Titan Image Generator G1.
    model_id = "amazon.titan-image-generator-v1"

    # Generate a random seed.
    seed = random.randint(0, 2147483647)

    # Prepare the request
    native_request = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 8.0,
            "height": 512,
            "width": 512,
            "seed": seed,
        },
    }

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=json.dumps(native_request))

        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract the image data.
        base64_image_data = model_response["images"][0]

        # Save the generated image to a local folder.
        i, output_dir = 1, "miridih-test/langgraph/img_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        while os.path.exists(os.path.join(output_dir, f"titan_{i}.png")):
            i += 1

        image_data = base64.b64decode(base64_image_data)

        image_path = os.path.join(output_dir, f"titan_{i}.png")
        with open(image_path, "wb") as file:
            file.write(image_data)

        print(f"The generated image has been saved to {image_path}")

    except ClientError as e:
        print(f"An error occurred: {e}")
        return None
