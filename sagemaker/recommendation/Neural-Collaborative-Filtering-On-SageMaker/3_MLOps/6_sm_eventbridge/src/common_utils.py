import json

def save_json(model_config_file_path, model_config_dict):
    '''
    해당 파일에 Json 저장
    '''
    model_config_json = json.dumps(model_config_dict)
    with open(model_config_file_path, "w") as outfile:
        outfile.write(model_config_json)   
    print(f"{model_config_file_path} is saved")    
        
    return model_config_file_path

def load_json(model_config_file_path):
    '''
    해당 경로의 json 로딩
    '''
    with open(model_config_file_path) as json_file:
        model_config_dict = json.load(json_file)
        
    return model_config_dict

    