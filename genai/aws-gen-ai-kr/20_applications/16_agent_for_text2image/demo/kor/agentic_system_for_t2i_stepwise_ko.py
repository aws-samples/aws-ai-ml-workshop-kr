import os
import io
import time
import json
import random
import pprint
import base64
import traceback
from PIL import Image
from termcolor import colored
import matplotlib.pyplot as plt

from textwrap import dedent
from utils.bedrock import bedrock_utils
from typing import TypedDict, Optional, List
from src.genai_anaysis import llm_call
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from utils.common_utils import retry
import boto3
from botocore.exceptions import ClientError, ConnectionError, ConnectTimeoutError, ReadTimeoutError

class TimeMeasurement:
    def __init__(self):
        self.start_time = None
        self.measurements = {}

    def start(self):
        self.start_time = time.time()

    def measure(self, section_name):
        if self.start_time is None:
            raise ValueError("start() 메서드를 먼저 호출해야 합니다.")
        
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.measurements[section_name] = elapsed_time
        self.start_time = end_time  # 다음 구간 측정을 위해 시작 시간 재설정

    def reset(self, ):
        self.measurements = {}

    def print_measurements(self):
        for section, elapsed_time in self.measurements.items():
            #print(f"{section}: {elapsed_time:.5f} 초")
            print(colored (f"\nelapsed time: {section}: {elapsed_time:.5f} 초", "red"))

# Set up Agent State
class GraphState(TypedDict):
    ask: str
    
    total_steps: int
    steps: List[str]
    ko_steps: List[str]

    seed: int
    current_step: int
    condition_image: str

    suggestions: str
    ko_suggestions: str
    prompt_repo: dict
    ko_prompt_repo: dict
    retry_count: int
    prev_node: str
    retouch: str
    should_regeneration: str

    error: bool
    error_message: str

# Build Agent for Text to Image
class genai_analyzer():

    def __init__(self, **kwargs):

        self.llm=kwargs["llm"]
        self.image_generation_model = kwargs["image_generation_model"]
        self.state = GraphState

        self.llm_caller = llm_call(
            llm=self.llm,
            verbose=False
        ) 

        self._graph_definition()
        self.messages = []

        self.timer = TimeMeasurement()

    def _get_string_from_message(self, message):
        return message["content"][0]["text"]

    def _get_message_from_string(self, role, string, imgs=None):
        
        message = {
            "role": role,
            "content": []
        }
        
        if imgs is not None:
            for img in imgs:
                img_message = {
                    "image": {
                        "format": 'png',
                        "source": {"bytes": img}
                    }
                }
                message["content"].append(img_message)
        
        message["content"].append({"text": dedent(string)})

        return message
    
    def _png_to_bytes(self, file_path):
        try:
            with open(file_path, "rb") as image_file:
                # 파일을 바이너리 모드로 읽기
                binary_data = image_file.read()
                
                # 바이너리 데이터를 base64로 인코딩
                base64_encoded = base64.b64encode(binary_data)
                
                # bytes 타입을 문자열로 디코딩
                base64_string = base64_encoded.decode('utf-8')
                
                return binary_data, base64_string
                
        except FileNotFoundError:
            return "Error: 파일을 찾을 수 없습니다."
        except Exception as e:
            return f"Error: {str(e)}"
        

    def show_save_image(self, base64_string, current_step, retry_count):
        try:
            
            # base64 문자열을 디코딩하여 바이너리 데이터로 변환
            image_data = base64.b64decode(base64_string)
            
            # 바이너리 데이터를 이미지로 변환
            image = Image.open(io.BytesIO(image_data))
            
            # save images
            img_path = f'{self.file_path_name}/GENERATED_IMAGE_{current_step}_{retry_count}.png'
            dir_path = os.path.dirname(img_path)  # Extract the directory path

            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            
            image.save(img_path, "PNG")
            time.sleep(3)
            
            return img_path
            
        except Exception as e:
            print(f"Error: 이미지를 표시하는 데 실패했습니다. {str(e)}")

    def _translate_text(self, text):
        # Create a Translate client
        translate = boto3.client('translate')
        
        try:
            # Call Amazon Translate to translate the text
            response = translate.translate_text(
                Text=text,
                SourceLanguageCode='en',  # English 
                TargetLanguageCode='ko'   # Korean
            )
            
            # Get the translated text from the response
            translated_text = response['TranslatedText']
            # print(translated_text)

            return translated_text
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return None
            
    def _body_generator(self, pos_prompt, neg_prompt="", condition_image=None, control_strength=None, seed=1):
        
        
        print ("_body_generator, control_strength", control_strength)
    
        if condition_image == None:
            self.image_generation_model.model_id = "stability.stable-image-ultra-v1:1"
            print (f'Image generator: SD-Ultra')
            body_dict = {
                "prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "mode": "text-to-image",
                "aspect_ratio": "3:2",  # Default 1:1. Enum: 16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21.
                "output_format": "png",
                "seed": seed
            }
        else:
            self.image_generation_model.model_id = "stability.sd3-5-large-v1:0"
            print (f'Image generator: SD-3-5-Large')
            body_dict = {
                "prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "mode": "image-to-image",
                "strength": control_strength, # nova랑 반대
                "image": condition_image,
                "output_format": "png",
                "seed": seed
            }

        return json.dumps(body_dict)

    def get_messages(self, ):
        return self.messages
        
    def _graph_definition(self, **kwargs):

        def StepwiseTaskDecomposer(state):

            self.timer.start()
            self.timer.reset()
            
            print("---StepwiseTaskDecomposer---")
            ask = state["ask"]
            current_step = state.get("current_step", 1)
            messages = []
        
            system_prompts = dedent(
                
                '''
                You are an agent that plans steps for stepwise image generation based on user requests.

                Core Responsibilities:
                
                1. Break down user requests into manageable steps that:
                   - Prioritize single step generation when feasible
                   - Only split into multiple steps (1-3) when complexity demands it
                     (e.g. layered scenes, multiple focal points, complex interactions)
                   - Follow control mode restrictions (NONE for step 1, SEGMENTATION after)
                   - Progress from core elements to details
                   - Use appropriate control strength for smooth transitions
                   - Track and maintain key subject and image style properties:
                     * Exact counts (e.g., "3 boats with passengers" not just "3 boats")
                     * Spatial orientations (e.g., "facing left")
                     * Specific attributes (e.g., "red cars" not just "cars")
                     * Relationships between subjects (e.g., "person sitting in each boat")
                     * Image style (e.g., "oil painting")

                2. For each step, provide:
                   - Start with new elements in prompt generation
                   - Step description
                   - Image generation prompt that have to maintain key subjects(people, car, etc) and elements from previous steps while clearly specifying new additions
                   - Add detailed improvements (style, lighting etc)
                   - Control mode (NONE/SEGMENTATION)
                   - Control strength (0.0-1.0, N/A for step 1)

                Step Planning Guidelines:
                First Step (Composition & Subject):
                - Uses NONE control mode
                - No control strength applicable
                - Must establish:
                  * Overall scene composition
                  * Main subjects and objects
                  * Spatial relationships and viewpoint
                  * Foreground/background structure
                  * Space allocation for future elements

                Subsequent Steps:
                - Uses SEGMENTATION control mode
                - Each prompt should explicitly reference maintaining previous subjects and elements
                - Control strength: 0.8-0.95 recommended
                - Consider new elements' impact when selecting control strength
                
                Scene Composition Rules:
                1. Foreground/Background
                   - Specify clear spatial relationships
                   - Maintain distinct layering
                   - Use explicit positioning terms

                2. Spatial Relationships
                   - Use clear position indicators (left, right, near, far)
                   - Consider depth and perspective
                   - Be explicit about distances and relationships

                Prompt Writing Guidelines:
                - Use image captioning style
                - Start with new elements or sbjects
                - Maintain consistent style across steps
                - Use clear, simple language
                - Keep under 5,000 characters
                - Include:
                  * Spatial relationships
                  * Depth indicators
                  * Viewing angles when relevant
                  * Style keywords at end

                Output Format:
                DO NOT include any text or json symbol (```json```)outside the JSON format in the response
                You must provide your response in the following JSON format:
                {
                    "total_steps": <number_of_steps>,
                    "steps": [
                        {
                            "step_number": <number>,
                            "description": <string>,
                            "control_mode": <"NONE"/"SEGMENTATION">,
                            "control_strength": <float>,
                            "prompt": {
                                "positive": <string>,
                                "negative": <string>
                            }
                        }
                    ]
                }

                Key Requirements:
                - Each step builds on previous
                - Maintain style consistency across steps through:
                  * Matching artistic style keywords
                  * Consistent quality enhancers
                  * Uniform lighting/atmosphere descriptions
                  * Consistent camera/perspective terms
                
                '''
            )

            if current_step == 1:
                system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts)
                user_prompts = dedent(
                    '''
                    Here is user's ask: <ask>{ask}</ask>
                    '''
                )
                context = {
                    "ask": ask,
                }
                user_prompts = user_prompts.format(**context)
                           
                message = self._get_message_from_string(role="user", string=user_prompts)
                self.messages.append(message)
                messages.append(message)
                
                resp, ai_message = self.llm_caller.invoke(
                    messages=messages,
                    system_prompts=system_prompts,
                    enable_reasoning=False,
                    reasoning_budget_tokens=1024
                )
                self.messages.append(ai_message)
                            
                results = json.loads(resp['text'])
                total_steps, steps = results["total_steps"], results["steps"]
                should_next_step = "next_step"

                # return self.state(
                #     total_steps=total_steps,
                #     steps=steps,
                #     should_next_step=should_next_step,
                #     prev_node="StepwiseTaskDecomposer"
                # )

                ko_steps = []
                for step in steps:
                    ko_step = {}
                    ko_step["prompt"] = {}
                    ko_step["step_number"] = step.get("step_number")
                    ko_step["control_mode"] = step.get("control_mode", "N/A")
                    ko_step["control_strength"] = step.get("control_strength", "N/A")
                    ko_step["description"] = self._translate_text(step.get("description"))
                    ko_step["prompt"]["positive"] = self._translate_text(step["prompt"]["positive"])
                    ko_step["prompt"]["negative"] = self._translate_text(step["prompt"]["negative"])
                    ko_steps.append(ko_step)
            
                
                return self.state(
                        total_steps=total_steps,
                        steps=steps,
                        ko_steps=ko_steps,
                        should_next_step=should_next_step,
                        prev_node="StepwiseTaskDecomposer"
                )
            
            else:
                generation_steps = state["steps"]
                if current_step <= len(generation_steps):
                    print ("---GO TO IMAGE GENERATION---")
                    print ("current_step: ", current_step)
                    should_next_step = "next_step"
                else:
                    should_next_step = "completed"
                    
                return self.state(
                    should_next_step=should_next_step,
                    prev_node="Reflection"
                )          
            
        def ShouldStepwiseImageGeneration(state):

            print("---ShouldStepwiseImageGeneration---")
            return state["should_next_step"]
        
        @retry(total_try_cnt=5, sleep_in_sec=60, retryable_exceptions=(ClientError,))
        def ImageGeneration(state):
            
            print("---ImageGeneration---")
            generation_steps, current_step = state["steps"], state.get("current_step", 1)
            retry_count = state.get("retry_count", 0)
            condition_image = state.get("condition_image", None)
            seed = state.get("seed", 1)
            prev_node = state.get("prev_node", None)
            
            pos_prompt = generation_steps[current_step-1]["prompt"]["positive"]
            neg_prompt = generation_steps[current_step-1]["prompt"]["negative"]
            control_mode = generation_steps[current_step-1]["control_mode"].upper()
            control_strength = generation_steps[current_step-1]["control_strength"]
            if prev_node == "PromptReformulation": seed = random.randint(0, 100000)

            seed = random.randint(0, 100000)
            print ("current_step", current_step)
            print ("retry_count", retry_count)
            print ("condition_image", condition_image)
            print ("prev_node", prev_node)
            print ("seed", seed)
                      
            if condition_image is not None: #and current_step != 1:
                img_bytes, img_base64 = self._png_to_bytes(condition_image)
                condition_image = img_base64
            else:
                condition_image = None
            
            body = self._body_generator(
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                condition_image=condition_image,
                control_strength=control_strength, # nova랑 반대
                seed=seed
            )
            
            response = self.image_generation_model.bedrock_client.invoke_model(
                body=body,
                modelId=self.image_generation_model.model_id
            )
            response_body = json.loads(response.get("body").read())
            base64_image = response_body.get("images")[0]
            condition_image = self.show_save_image(base64_image, current_step, retry_count)
            
            return self.state(
                condition_image=condition_image,
                current_step=current_step,
                prev_node="ImageGeneration"
            )
        
        def PromptReformulation(state):
            
            print("---PromptReformulation---")
            generation_steps = state["steps"]
            suggestions = state["suggestions"]
            current_step = state["current_step"]
            retry_count = state.get("retry_count", 0)

            pos_prompt = generation_steps[current_step-1]["prompt"]["positive"]
            neg_prompt = generation_steps[current_step-1]["prompt"]["negative"]
            original_prompt = f'positive: {pos_prompt}, negative: {neg_prompt}'
            messages=[]
            
            system_prompts = dedent(
                '''
                You are an agent that enhances image generation prompts based on provided suggestions. Your role is to:

                1. Process Input:
                   - Original image generation prompt
                   - Provided suggestions for improvement

                2. Enhance Prompt:
                   - Start with new elements or sbjects
                   - Maintain the core elements and structure of original prompt
                   - Keep the total prompt length under 5,000 characters
                   - Write prompts as concisely as possible
                   - 제거되어야 하는 사항이 있다면 "negative" prompt에 넣어 주세요. 
                   
                3. Determine Control Strength:
                   - 0.8-0.95: Optimal range for balanced transformation
                   - A value of 0 would yield an image that is identical to the input. A value of 1 would be as if you passed in no image at all. Range: [0, 1]
                   - Consider the impact on existing elements

                Required Output Format:
                DO NOT include any text or json symbol (```json```)outside the JSON format in the response
                You must ONLY output the JSON object, nothing else.
                NO descriptions of what you’re doing before or after JSON.
                Note: “control_strength” must be included inside the “prompt_repo” object.
                {
                    “prompt_repo”: {
                        “positive”: <improved prompt incorporating suggestions>,
                        “negative”: <negative prompt>,
                        “control_strength”: <float between 0.0 and 1.0>
                    }
                }
                
                General Guidelines:
                - Keep the original prompt's main structure
                - Integrate suggestions naturally
                - Use image captioning style
                - Maintain clear spatial relationships
                - Ensure coherent flow in descriptions
                - Preserve essential elements from original prompt
                - Use concise, clear descriptions
                - Prioritize critical elements when length is constrained
                - Remove redundant or unnecessary descriptors
                - Stay within 5,000 character limit
                - Ensure style consistency with previous steps
                
                '''
            )

            system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts)
            user_prompts = dedent(
                '''
                Here is original prompt: <original_prompt>{original_prompt}</original_prompt>
                Here is suggestions: <suggestions>{suggestions}</suggestions>
                '''
            )
            context = {
                "original_prompt": original_prompt,
                "suggestions": suggestions
            }
            user_prompts = user_prompts.format(**context)
                       
            message = self._get_message_from_string(role="user", string=user_prompts)
            self.messages.append(message)
            messages.append(message)
            
            resp, ai_message = self.llm_caller.invoke(
                messages=messages,
                system_prompts=system_prompts
            )
            self.messages.append(ai_message)
                        
            results = json.loads(resp['text'])
            prompt_repo = results["prompt_repo"]

            print ("=================before")

            print ("pos:", generation_steps[current_step-1]["prompt"]["positive"])
            print ("neg:", generation_steps[current_step-1]["prompt"]["negative"])
            print ("control_strength:", generation_steps[current_step-1]["control_strength"])

            try:
                generation_steps[current_step-1]["prompt"]["positive"] = prompt_repo["positive"]
                generation_steps[current_step-1]["prompt"]["negative"] = prompt_repo["negative"]
                generation_steps[current_step-1]["control_strength"] = prompt_repo["control_strength"]

                print ("=================after")
                print ("pos:", generation_steps[current_step-1]["prompt"]["positive"])
                print ("neg:", generation_steps[current_step-1]["prompt"]["negative"])
                print ("control_strength:", generation_steps[current_step-1]["control_strength"])

                ko_prompt_repo = {}
                ko_prompt_repo["positive"] = self._translate_text(prompt_repo["positive"])
                ko_prompt_repo["negative"] = self._translate_text(prompt_repo["negative"])
                ko_prompt_repo["control_strength"] = prompt_repo["control_strength"]

                return self.state(
                    error=False,
                    current_step=current_step,
                    generation_steps=generation_steps,
                    prompt_repo=prompt_repo,
                    ko_prompt_repo=ko_prompt_repo,
                    prev_node="PromptReformulation"
                )
            except KeyError as e:
                error_message = f"""
                    KeyError: {e}.\n
                    The LLM did not adhere to the specified output format. But, proceed anyway.
                    """
                print(error_message)
                return self.state(error=True, error_message=error_message)
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return self.state(error=True, error_message=e)
            
        
        def Reflection(state):
            
            print("---Reflection---")
            generation_steps = state["steps"]
            current_step = state["current_step"]
            condition_image = state["condition_image"]
            retry_count = state.get("retry_count", 0)

            pos_prompt = generation_steps[current_step-1]["prompt"]["positive"]
            neg_prompt = generation_steps[current_step-1]["prompt"]["negative"]
            step_ask = f'positive: {pos_prompt}, negative: {neg_prompt}'
            messages = []
            
            print ("step_ask", step_ask)
        
            system_prompts = dedent(
                '''
                You are an image quality evaluator.
                Compare the generated image with the user's requirements and provide an assessment focusing on accuracy and alignment.
                Evaluate whether all requested elements are present and match the requirements.
                
                Output your evaluation in the following JSON format:
                DO NOT include any text or json symbol (```json```)outside the JSON format in the response
                You must ONLY output the JSON object, nothing else.
                NO descriptions of what you're doing before or after JSON.
                {
                    "retouch": "true/false",  // true if elements don't match (MUST mark true for ANY mismatch in counts, positions, or orientations of subjects/objects)
                    "suggestions": [
                        "list mismatches first"
                    ],
                    "evaluation": {
                        "key_subjects": {
                            "subject_name": {
                                "results": "match/mismatch",
                                "count": "actual count",
                                "attributes": "key details"
                            }
                        },
                        "composition": {
                            "alignment": "evaluation of layout and positioning"
                        },
                        "style": {
                            "overall": "evaluation of style and mood"
                        }
                    }
                }
                Provide clear, concise feedback for any mismatches.
               
                '''
             )

            system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts)
            user_prompts = dedent(
                '''
                Here is the user requests: <user_requests>{ask}</user_requests>
                '''
            )    
            context = {
                "ask": step_ask
            }
            user_prompts = user_prompts.format(**context)
            
            img_bytes, img_base64 = self._png_to_bytes(condition_image)
            message = self._get_message_from_string(role="user", string=user_prompts, imgs=[img_bytes])
            messages.append(message)
            self.messages.append(message)

            resp, ai_message = self.llm_caller.invoke(
                messages=messages,
                system_prompts=system_prompts
            )
            self.messages.append(ai_message)

            results = json.loads(resp['text'])
            suggestions = results["suggestions"]
            retouch, suggestions = results["retouch"], results["suggestions"]
            
            ko_suggestions = []
            for suggestion in suggestions:
                ko_suggestions.append(self._translate_text(suggestion))

            if retouch == "true":
                retry_count += 1
                if retry_count <= 2: should_regeneration = "retouch"
                else:
                    retry_count = 0
                    current_step += 1
                    should_regeneration = "pass"
            else:
                retry_count = 0  
                current_step += 1
                should_regeneration = "pass"

            return self.state(
                retouch=retouch,
                suggestions=suggestions,
                ko_suggestions=ko_suggestions,
                retry_count=retry_count,
                current_step=current_step,
                should_regeneration=should_regeneration,
                prev_node="Reflection"
            )
            messages = []

        def ShouldImageRegeneration(state):
            
            print("---ShouldImageRegeneration---")

            return state["should_regeneration"]

        # langgraph.graph에서 StateGraph와 END를 가져옵니다.
        workflow = StateGraph(self.state)

        # Todo 를 작성합니다.
        workflow.add_node("StepwiseTaskDecomposer", StepwiseTaskDecomposer)  # 이미지 생성을 위해 필요한 요소들이 준비되었는지 확인합니다.
        workflow.add_node("ImageGeneration", ImageGeneration)  # 요청을 이미지 생성용 프롬프트로 수정하는 노드를 추가합니다.
        workflow.add_node("Reflection", Reflection)  # 사용자의 요청에 맞게 이미지가 생성 되었는지 확인힙니다.
        workflow.add_node("PromptReformulation", PromptReformulation)  # 사용자의 요청에 맞게 이미지가 생성 되었는지 확인힙니다.
        
        workflow.add_conditional_edges(
            "StepwiseTaskDecomposer",
            # 에이전트 결정 평가
            ShouldStepwiseImageGeneration,
            {
                # 도구 노드 호출
                "next_step": "ImageGeneration",
                "completed": END,
            },
        )
        workflow.add_edge("ImageGeneration", "Reflection")
        workflow.add_conditional_edges(
            "Reflection",
            # 에이전트 결정 평가
            ShouldImageRegeneration,
            {
                # 도구 노드 호출
                "pass": "StepwiseTaskDecomposer",
                "retouch": "PromptReformulation"
            },
        )
        workflow.add_edge("PromptReformulation", "ImageGeneration")
        
        # 시작점을 설정합니다.
        workflow.set_entry_point("StepwiseTaskDecomposer")

        # 기록을 위한 메모리 저장소를 설정합니다.
        memory = MemorySaver()

        # 그래프를 컴파일합니다.
        self.app = workflow.compile(checkpointer=memory)        
        self.config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "Text2Image"})

    def invoke(self, **kwargs):
        
        inputs = self.state(
            ask=kwargs["ask"],
            image_model=kwargs["image_model"]
        )
        self.file_path_name = kwargs.get('file_path_name')
        
        # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.
        for output in self.app.stream(inputs, self.config):
            # 출력된 결과에서 키와 값을 순회합니다.
            for key, value in output.items():
                # 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
                pprint.pprint(f"\nOutput from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value, indent=2, width=80, depth=None)

                yield {"key": key, "value": value}
                
            # 각 출력 사이에 구분선을 추가합니다.
            pprint.pprint("\n---\n")