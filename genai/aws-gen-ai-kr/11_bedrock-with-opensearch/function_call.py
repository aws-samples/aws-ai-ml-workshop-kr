import requests
import sys
from defusedxml import ElementTree
from collections import defaultdict
import os
from typing import Any
import boto3
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


class Weathertools:
    # To add a tool to be used by Claude in main_demo.py,
    # create your tool in python as shown below and then create
    # a new string variable describing the tool spec. Copy the XML formatting
    # that is shown in the below example.
    #
    # Once you have created your tool and your spec, add the spec variable to the 
    # list_of_tools_specs list.
    
    def __init__(self):
        self.get_lat_long_description = \
        """<tool_description>
            <tool_name>get_lat_long</tool_name>
            <description>
            Returns the latitude and longitude for a given place name.
            </description>
            <parameters>
            <parameter>
            <name>place</name>  
            <type>string</type>
            <description>
            The place name to geocode and get coordinates for.
            </description>
            </parameter>
            </parameters>
            </tool_description>"""
        
        self.get_weather_description = \
        """
            <tool_description>
            <tool_name>get_weather</tool_name>
            <description>
            Returns weather data for a given latitude and longitude. </description>
            <parameters>
            <parameter>
            <name>latitude</name>
            <type>string</type>
            <description>The latitude coordinate as a string</description>
            </parameter> <parameter>
            <name>longitude</name>
            <type>string</type>
            <description>The longitude coordinate as a string</description>
            </parameter>
            </parameters>
            </tool_description>
            """       
        self.list_of_tools_specs = [self.get_weather_description, self.get_lat_long_description]
        

    def get_weather(self, latitude: str, longitude: str):
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        response = requests.get(url)
        return response.json()



    def get_lat_long(self, place):
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': place, 'format': 'json', 'limit': 1}
        response = requests.get(url, params=params).json()

        if response:
            lat = response[0]["lat"]
            lon = response[0]["lon"]
            return {"latitude": lat, "longitude": lon}
        else:
            return None

tools = Weathertools()

def add_tools():
    
    tools_string = ""
    for tool_spec in tools.list_of_tools_specs:
        tools_string += tool_spec
    return tools_string


def call_function(tool_name, parameters):
    func = getattr(tools, tool_name)
    output = func(**parameters)
    return output


def format_result(tool_name, output):
    return f"""<function_results>
                    <result>
                        <tool_name>{tool_name}</tool_name>
                        <stdout>{output}</stdout>
                    </result>
                </function_results>"""


def etree_to_dict(t) -> dict[str, Any]:
    d = {t.tag: {}}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text and t.text.strip():
        if children or t.attrib:
            d[t.tag]["#text"] = t.text
        else:
            d[t.tag] = t.text
    return d


def run_loop(prompt, region_name):
    print(prompt)
    # Start function calling loop
    while True:
    # initialize variables to make bedrock api call
        bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

        prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature" : 0,
            "top_k": 350,
            "top_p": 0.999,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "stop_sequences":["\n\nHuman:", "</function_calls>"]
        }


        body = json.dumps(prompt_config)


        # bedrock api call with prompt
        response = bedrock.invoke_model(
            # body=json.dumps(
            #     {
            #         "prompt": prompt,
            #          "stop_sequences":["\n\nHuman:", "</function_calls>"],
            #          "max_tokens_to_sample": 700,
            #          "temperature": 0
            #     }
            # ), 
            body=body,
            modelId="anthropic.claude-3-haiku-20240307-v1:0", 
            accept='application/json', 
            contentType='application/json'
        )
        response_body = json.loads(response.get("body").read())
        completion = response_body.get("content")[0].get("text")

        stop_reason = response_body.get('stop_reason')
        stop_seq = completion.rstrip().endswith("</invoke>")
        
        # Get a completion from Claude

        # Append the completion to the end of the prommpt
        prompt += completion
        
        if stop_reason == 'stop_sequence' and stop_seq:
            # If Claude made a function call
            # print(completion)
            start_index = completion.find("<function_calls>")
            if start_index != -1:
                # Extract the XML Claude outputted (invoking the function)
                extracted_text = completion[start_index+16:]

                # Parse the XML find the tool name and the parameters that we need to pass to the tool
                xml = ElementTree.fromstring(extracted_text)
                tool_name_element = xml.find("tool_name")
                
                if tool_name_element is None:
                    print("Unable to parse function call, invalid XML or missing 'tool_name' tag")
                    break
                tool_name_from_xml = tool_name_element.text.strip()
                parameters_xml = xml.find("parameters")
                if parameters_xml is None:
                    print("Unable to parse function call, invalid XML or missing 'parameters' tag")
                    break
                param_dict = etree_to_dict(parameters_xml)
                parameters = param_dict["parameters"]

                # Call the tool we defined in tools.py
                output = call_function(tool_name_from_xml, parameters)

                # Add the stop sequence back to the prompt
                prompt += "</function_calls>"
                print("</function_calls>")

                # Add the result from calling the tool back to the prompt
                function_result = format_result(tool_name_from_xml, output)
                print(function_result)
                prompt += function_result
        else:
            # If Claude did not make a function call
            # outputted answer
            print(completion)
            break