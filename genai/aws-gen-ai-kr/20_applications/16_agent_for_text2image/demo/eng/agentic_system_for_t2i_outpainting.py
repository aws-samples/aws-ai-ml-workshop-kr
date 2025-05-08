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



class GraphState(TypedDict):
    ask: str
    task_type: str
    ask_repo: str
    origin_ask_repo: str
    prompt_components: dict
    image_prompt: dict
    image_model: str
    generated_img_path: str
    suggestions: str
    retouch: str
    retry_count: int
    control_image_needed: str
    control_mode: str
    mask_image: str
    prev_node: str
    original_image: str

    error: bool
    error_message: str

class GenAIOutPainting():
    """A class for editing images."""
    def __init__(self, **kwargs):
        """Initialize the GenAIOutPainting with required models and components.
        
        Args:
            **kwargs: Keyword arguments including llm and image_generation_model.
        """
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
        """Extract text content from a message.
        
        Args:
            message: A message dictionary containing content.
            
        Returns:
            str: The text content from the message.
        """
        return message["content"][0]["text"]

    def _get_message_from_string(self, role, string, imgs=None):
        """Create a message dictionary from text and optional images.
        
        Args:
            role: The role of the message sender.
            string: The text content.
            imgs: Optional list of images to include.
            
        Returns:
            dict: A formatted message dictionary.
        """
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
        """Convert a PNG file to binary data and base64 string.
        
        Args:
            file_path: Path to the PNG file.
            
        Returns:
            tuple: (binary_data, base64_string) or error message.
        """
        try:
            with open(file_path, "rb") as image_file:
                # Read file in binary mode
                binary_data = image_file.read()
                
                # Encode binary data to base64
                base64_encoded = base64.b64encode(binary_data)
                
                # Decode bytes to string
                base64_string = base64_encoded.decode('utf-8')
                
                return binary_data, base64_string
                
        except FileNotFoundError:
            return "Error: 파일을 찾을 수 없습니다."
        except Exception as e:
            return f"Error: {str(e)}"

    def show_save_image(self, base64_string):
        """Display and save an image from base64 string.
        
        Args:
            base64_string: Base64 encoded image data.
            
        Returns:
            str: Path to the saved image file.
        """
        try:
            
            # Decode base64 string to binary data
            image_data = base64.b64decode(base64_string)
            
            # Convert binary data to image
            image = Image.open(io.BytesIO(image_data))

            # save images
            img_path = self.file_name
            dir_path = os.path.dirname(img_path)
            
            image.save(img_path, "PNG")
            
            return img_path
            
        except Exception as e:
            print(f"Error: 이미지를 표시하는 데 실패했습니다. {str(e)}")
            
    def _body_generator(
            self, image_prompt, taskType="OUTPAINTING", maskImage=None,
            original_image=None,
    ):    
        """Generate request body for image generation API.
        
        Args:
            image_prompt: Dictionary containing main_prompt and negative_prompt.
            taskType: Type of image generation task.
            maskImage: Optional mask image for outpainting.
            original_image: Optional original image for outpainting.
            
        Returns:
            str: JSON string for the request body.
            
        Raises:
            ValueError: If required parameters are missing for specific task types.
        """
        if taskType == "OUTPAINTING":
            # Create request body for outpainting
            if maskImage is not None:
                # Read mask image file
                _, mask_base64 = self._png_to_bytes(maskImage)
                _, img_base64 = self._png_to_bytes(original_image)
                
                body_dict = {
                    "taskType": "OUTPAINTING",
                    "outPaintingParams": {
                        "image": img_base64,  # Original image
                        "maskImage": mask_base64,  # Mask image
                        "text": image_prompt["main_prompt"],
                        "outPaintingMode": "PRECISE",
                        "negativeText": image_prompt["negative_prompt"]
                    },
                    "imageGenerationConfig": {
                        "numberOfImages": 1,
                        "quality": "premium",
                        "cfgScale": 10,
                        "seed": 12,
                    }
                }
            else:
                raise ValueError("maskImage는 OUTPAINTING 작업 유형에 필수입니다.")
        else:
            raise ValueError("유효하지 않은 taskType입니다. 'OUTPAINTING' 이어야 합니다.")
    
        return json.dumps(body_dict)

    def get_messages(self, ):
        return self.messages
        
    def _graph_definition(self, **kwargs):
        """Define the workflow graph for image generation.
        
        Args:
            **kwargs: Additional keyword arguments.
        """

        def ask_reformulation(state):
            """Reformulate user's ask into an optimized request.

            Args:
                state: Current state dictionary.
                
            Returns:
                Updated state with reformulated ask.
            """

            self.timer.start()
            self.timer.reset()
            
            print("---ASK REFORMULATION---")
            ask = state["ask"]
            image_prompt = state.get("image_prompt", "None")
            origin_ask_repo = state.get("origin_ask_repo", "None")
            messages = []
            
            print ("image_prompt", image_prompt)
            
            system_prompts = dedent(
                '''
                <task>
                당신은 GenAI 활용 Outpainting 전문가입니다. 사용자의 원본 요청(ask)을 바탕으로 아래의 요소를 추출하세요.
                main_prompt: "A text prompt that describes what to generate within the masked region. If you omit this field, the model will remove elements inside the masked area. They will be replaced with a seamless extension of the image background"
                negative_prompt: "A text prompt to define what not to include in the image."
                </task>
            
                <instruction>
                우선순위에 따라 요청을 체계적으로 재구성하세요:
                1. 사용자의 원본 요청(ask)이 최우선 고려사항입니다.
                2. 모든 요소들이 자연스럽게 통합되도록 조정하세요.
            
                </instruction>
            
                <output_format>
                반드시 다음 형식의 JSON만 반환하고 설명이나 추가 텍스트를 포함하지 마세요:
                
                {
                    "ask_repo": "재구성된 요청",
                }
                </output_format>
                '''
            )
            
            system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts)
            user_prompts = dedent(
                '''
                Here is user's ask: <ask>{ask}</ask>
                Here is main_prompt: <main_prompt>{main_prompt}</main_prompt>
                Here is negative_prompt: <negative_prompt>{negative_prompt}</negative_prompt>
                '''
            )
            context = {
                "ask": ask,
                "main_prompt": image_prompt["main_prompt"] if image_prompt != "None" else "None",
                "negative_prompt": image_prompt["negative_prompt"] if image_prompt != "None" else "None",
            }
            user_prompts = user_prompts.format(**context)
                       
            message = self._get_message_from_string(role="user", string=user_prompts)
            self.messages.append(message)
            messages.append(message)
            
            resp, ai_message = self.llm_caller.invoke(messages=messages, system_prompts=system_prompts)
            self.messages.append(ai_message)
            results = eval(resp['text'])
            ask_repo = results["ask_repo"]
            
            if origin_ask_repo == "None": origin_ask_repo = ask_repo
            
            return self.state(
                ask_repo=ask_repo,
                origin_ask_repo=origin_ask_repo,
                prev_node="ASK_REFORMULATION")   
            
        def check_readness_prompt_generation(state):
            """Check readiness for prompt generation by extracting visual components.
            
            Args:
                state: Current state dictionary.
                
            Returns:
                Updated state with extracted prompt components.
            """
            print("---CHECK READNESS FOR PROMPT GENERATION---")
            ask_repo = state["ask_repo"]
            image_prompt = state.get("image_prompt", "None")
            messages = []
            print ("image_prompt", image_prompt)
            
            system_prompts = dedent(
                '''
                <task>
                사용자의 이미지 생성 요청을 분석하여 6가지 핵심 시각적 요소를 정확히 추출하세요.
                </task>
            
                <instruction>
                다음 핵심 시각 요소들을 사용자 요청에서 식별하고 정확히 추출하세요:
            
                1. subject (주체): 이미지의 중심이 되는 인물, 물체 또는 개체
                   - 예: "여자", "고양이", "산", "도시 풍경"
            
                2. action (행동): 주체가 취하는 동작이나 상태
                   - 예: "달리는", "웃고 있는", "떠오르는", "휴식 중인"
            
                3. environment (환경): 배경 장소나 주변 환경
                   - 예: "해변", "도시", "우주", "숲속"
            
                4. lighting (조명): 빛의 상태나 분위기
                   - 예: "일몰", "푸른 빛", "어두운", "밝고 화창한"
            
                5. style (스타일): 예술적 표현 방식이나 참조
                   - 예: "수채화", "사실적", "애니메이션", "미니멀리즘"
            
                6. camera_position (카메라 위치): 시점이나 프레이밍
                   - 예: "클로즈업", "조감도", "측면 각도", "넓은 샷"
            
                각 요소가 요청에 명시적으로 언급되지 않은 경우 해당 필드는 None로 설정하세요.
                관련 내용이 있다면 사용자의 원문을 최대한 그대로 추출하세요.
                암시적으로 언급된 요소도 파악하여 추출하세요.
                </instruction>
            
                <output_format>
                반드시 다음 형식의 JSON만 반환하고 설명이나 추가 텍스트를 포함하지 마세요:
                
                {
                    "components": {
                        "subject": {
                            "content": "발견된 텍스트 또는 None"
                        },
                        "action": {
                            "content": "발견된 텍스트 또는 None"
                        },
                        "environment": {
                            "content": "발견된 텍스트 또는 None"
                        },
                        "lighting": {
                            "content": "발견된 텍스트 또는 None"
                        },
                        "style": {
                            "content": "발견된 텍스트 또는 None"
                        },
                        "camera_position": {
                            "content": "발견된 텍스트 또는 None"
                        }
                    }
                }
                </output_format>
                '''
            )
                
            system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts)
            user_prompts = dedent(
                '''
                Here is user's ask: <ask>{ask}</ask>
                Here is main_prompt: <main_prompt>{main_prompt}</main_prompt>
                Here is negative_prompt: <negative_prompt>{negative_prompt}</negative_prompt>
                '''
            )
            context = {
                "ask": ask_repo,
                "main_prompt": image_prompt["main_prompt"] if image_prompt != "None" else "None",
                "negative_prompt": image_prompt["negative_prompt"] if image_prompt != "None" else "None"                
            }
            user_prompts = user_prompts.format(**context)
                       
            message = self._get_message_from_string(role="user", string=user_prompts)
            self.messages.append(message)
            messages.append(message)
            
            resp, ai_message = self.llm_caller.invoke(messages=messages, system_prompts=system_prompts)
            self.messages.append(ai_message)
            results = eval(resp['text'])
            prompt_components = results["components"]
            
            return self.state(prompt_components=prompt_components, prev_node="CHECK_READNESS_PROMPT_GENERATION")
                
        def prompt_generation_for_image(state):
            """Generate optimized image prompts based on extracted components.
            
            Args:
                state: Current state dictionary.
                
            Returns:
                Updated state with generated image prompt.
            """
            print("---PROMPT GENERATION FOR IMAGE---")
            ask_repo, prompt_components, image_model,  = state["ask_repo"], state["prompt_components"], state["image_model"]
            image_prompt = state.get("image_prompt", "None")
            messages = []
            
            system_prompts = dedent(
                '''
                <task>
                추출된 시각 요소들을 활용하여 이미지 outpainting 모델 {image_model}에 최적화된 고품질 프롬프트를 생성하세요.
                </task>
            
                <instruction>
                이미지 outpainting 프롬프트 전문가로서, 다음 원칙에 따라 최적의 프롬프트를 구성하세요:
            
                1. 이미지 캡션 형태로 작성
                   - 명령문("~해줘", "~그려줘")이나 대화체 표현을 완전히 제거
                   - 모든 설명은 영어로 변환
                   - 묘사적이고 구체적인 명사구/형용사구 사용
            
                2. {image_model} 최적화 전략:
                   - nova-canvas 최적화:
                     * 구체적이고 정확한 시각적 설명
                     * 해상도, 렌더링 품질 관련 키워드 추가
                     * 세부 묘사를 중심으로 구성
            
                3. 프롬프트 구성 원칙:
                   - 중요 요소를 문장 앞쪽에 배치
                   - 콤마(,)로 구분하여 요소 간 가중치 균형 유지
                   - 핵심 시각적 요소에 대한 디테일 강화
                   - 부정 표현("no", "not", "without" 등)은 사용하지 말고 negative_prompt 필드에 배치
            
                4. 프롬프트 길이는 1024자 이내로 유지하세요.
                </instruction>
            
                <output_format>
                DO NOT include any text or json symbol (```json```)outside the JSON format in the response
                다음 형식의 JSON으로만 응답하세요:
                {{
                    "image_prompt": 
                    {{
                        "main_prompt": "재구성된 이미지 캡션 형태의 프롬프트",
                        "negative_prompt": "제외할 요소들"
                    }}
                }}
                </output_format>
                '''
            )
            
            context = {"image_model": image_model}
            system_prompts = system_prompts.format(**context)
            system_prompts = bedrock_utils.get_system_prompt(system_prompts=system_prompts)
            
            user_prompts = dedent(
                '''
                Here is user's ask: <ask>{ask}</ask>
                Here is extracted components: <subject>{subject}</subject>,\n<action>{action}</action>,\n<environment>{environment}</environment>\n<lighting>{lighting}</lighting>\n<style>{style}</style>\n<camera_position>{camera_position}</camera_position>
                '''
            )
            context = {
                "ask": ask_repo,
                "subject": prompt_components["subject"],
                "action": prompt_components["action"],
                "environment": prompt_components["environment"],
                "lighting": prompt_components["lighting"],
                "style": prompt_components["style"],
                "camera_position": prompt_components["camera_position"]
            }
            user_prompts = user_prompts.format(**context)
            
            message = self._get_message_from_string(role="user", string=user_prompts)            
            self.messages.append(message)
            messages.append(message)

            resp, ai_message = self.llm_caller.invoke(messages=messages, system_prompts=system_prompts)
            self.messages.append(ai_message)

            results = eval(resp['text'])
            image_prompt = results["image_prompt"]

            return self.state(image_prompt=image_prompt, prev_node="PROMPT_GENERATION_FOR_IMAGE")

        def image_generation(state):
            """Generate an image based on the prepared prompts.
            
            Args:
                state: Current state dictionary.
                
            Returns:
                Updated state with generated image path.
            """
            print("---IMAGE GENERATION---")
            image_prompt = state["image_prompt"]
            generated_img_path = state.get("generated_img_path", None)
            task_type = state.get("task_type", "OUTPAINTING")  # 추가: 기본값은 "OUTPAINTING"
            mask_image = state.get("mask_image", None)  # 추가: 마스크 이미지 파일 경로
            original_image = state.get("original_image", None)  # 추가: 마스크 이미지 파일 경로
            print("generated_img_path", generated_img_path)
            print("task_type", task_type)
            print("mask_image", mask_image)
            print("original_image", original_image)
        
            # 이미지 생성 요청 본문 생성
            body = self._body_generator(
                image_prompt,
                taskType=task_type,
                maskImage=mask_image,
                original_image=original_image, ## in/out painting 위해서
            )
            
            # 이미지 생성 API 호출
            # response = self.image_generation_model.bedrock_client.invoke_model(
            #     body=body,
            #     modelId=self.image_generation_model.model_id
            # )

            # 이미지 생성 API 호출
            try:
                response = self.image_generation_model.bedrock_client.invoke_model(
                    body=body,
                    modelId=self.image_generation_model.model_id
                )
            except ClientError as e:
                error_message = e.response['Error']['Message']
                print(f"Bedrock API error: {error_message}")
                return self.state(error_message=error_message, prev_node="IMAGE_GENERATION", error=True)
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return self.state(error_message=error_message, prev_node="IMAGE_GENERATION", error=True)
            
            response_body = json.loads(response.get("body").read())
            try:
                base64_image = response_body.get("images")[0]
                generated_img_path = self.show_save_image(base64_image)
                return self.state(generated_img_path=generated_img_path, prev_node="IMAGE_GENERATION", error=False)
            except Exception as e:
                print(f"Unexpected error: {response_body['error']}")
                return self.state(error_message=response_body['error'], prev_node="IMAGE_GENERATION", error=True)

        def should_image_regeneration(state):
            """Determine if image should be regenerated.
            
            Args:
                state: Current state dictionary.
                
            Returns:
                str: Decision on whether to regenerate the image.
            """
            print("---IMAGE CHECKER---")
            retouch, retry_count = state["retouch"], state["retry_count"]
            
            if retry_count <= 2 and retouch == "true":
                print ("---[REFLECTION] GO TO IMAGE REGENERATION---")
                print ("retry_count: ", retry_count)
                return "regeneration"
            else:
                print ("---GO TO SHOW UP---")
                return "continue"
            
        # langgraph.graph에서 StateGraph와 END를 가져옵니다.
        workflow = StateGraph(self.state)

        # Todo 를 작성합니다.
        workflow.add_node("ask_reformulation", ask_reformulation)  # 이미지 생성을 위해 필요한 요소들이 준비되었는지 확인합니다.
        workflow.add_node("check_readness_prompt_generation", check_readness_prompt_generation)  # 이미지 생성을 위해 필요한 요소들이 준비되었는지 확인합니다.
        workflow.add_node("prompt_generation_for_image", prompt_generation_for_image)  # 요청을 이미지 생성용 프롬프트로 수정하는 노드를 추가합니다.
        workflow.add_node("image_generation", image_generation)  # 이미지 생성하는 노드를 추가합니다.
        
        workflow.add_edge("ask_reformulation", "check_readness_prompt_generation")
        workflow.add_edge("check_readness_prompt_generation", "prompt_generation_for_image")
        workflow.add_edge("prompt_generation_for_image", "image_generation")
        workflow.add_edge("image_generation", END)
        
        # 시작점을 설정합니다.
        workflow.set_entry_point("ask_reformulation")

        # 기록을 위한 메모리 저장소를 설정합니다.
        memory = MemorySaver()

        # 그래프를 컴파일합니다.
        self.app = workflow.compile(checkpointer=memory)        
        self.config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "Text2Image"})

    def invoke(self, **kwargs):
        """Run the image generation workflow with the given inputs.
        
        Args:
            **kwargs: Input parameters including ask, image_model, 
                      and optional task_type, mask_image, and original_image.
                      
        Returns:
            None: Results are stored in class attributes.
        """
        
        # 새로운 매개변수 추가: task_type, mask_image
        inputs = self.state(
            ask=kwargs["ask"], 
            image_model=kwargs["image_model"],
            task_type=kwargs.get("task_type", "OUTPAINTING"),  # 기본값은 "OUTPAINTING"
            mask_image=kwargs.get("mask_image", None),  # 기본값은 None
            original_image=kwargs.get("original_image", None),  # 기본값은 None
        )
        self.file_name = kwargs.get('file_name')
       

        # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.
        for output in self.app.stream(inputs, self.config):
            # 출력된 결과에서 키와 값을 순회합니다.
            for key, value in output.items():
                # 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
                pprint.pprint(f"\nOutput from node '{key}':")
                pprint.pprint("---")
                # 출력 값을 예쁘게 출력합니다.
                pprint.pprint(value, indent=2, width=80, depth=None)

                yield {"key": key, "value": value}
                
            # 각 출력 사이에 구분선을 추가합니다.
            pprint.pprint("\n---\n")