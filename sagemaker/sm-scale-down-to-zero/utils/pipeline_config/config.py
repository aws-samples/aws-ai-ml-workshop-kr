import os
import ast
from os import path
from configparser import ConfigParser, ExtendedInterpolation

class Empty:
    pass

class config_handler():
    
    def __init__(self, strConfigPath="config.ini"):
        
        strFilePath = path.dirname(path.abspath(__file__)) # current directory
        self.parser = ConfigParser(interpolation=ExtendedInterpolation())
        self.parser.read(os.path.join(strFilePath, strConfigPath))
        self.get_all_info()
      
    def get_all_info(self, ):
        
        print ("====== config info. ======")
        for strSection in self.parser.sections():
            for strOption in self.parser.options(strSection):
                print (f"  {strSection}: {strOption}:{self.parser.get(strSection, strOption)}")
        print ("==========================")
        
    def get_value(self, strSection, strOption, dtype="str"):
        
        if dtype == "str": return self.parser.get(strSection, strOption)
        elif dtype == "int": return self.parser.getint(strSection, strOption)
        elif dtype == "float": return self.parser.getfloat(strSection, strOption)
        elif dtype == "boolean": return self.parser.getboolean(strSection, strOption)
        elif dtype in {"list", "dict"}: return ast.literal_eval(self.parser.get(strSection, strOption))

    def set_value(self, strSection, strOption, newValue):
        
        if not self.parser.has_section(strSection): self.parser.add_section(strSection)
        self.parser[strSection][strOption] = str(newValue)

        if not hasattr(self, strSection): setattr(self, strSection, Empty())
        current_section = getattr(self, strSection)
        setattr(current_section, strOption, newValue)
        
    def member_section_check(self, strSection):
        return self.parser.has_section(strSection)
    
    def member_key_check(self, strSection, strOption):
        return self.parser.has_option(strSection, strOption)

if __name__ == "__main__":
    
    

    config = config_handler()
    
    
    config.set_value("ROLE", "sd", "22")
    config.set_value("UPDATE", "new_key", "22")
    
    config.get_all_info()
    
    #inference_instances = ["ml.t2.medium", "ml.c5.4xlarge"] ## for string
    #inference_instances = [1, 2] ## for numeric
    
    A = config.get_value("MODEL_REGISTER", "inference_instances_", dtype="list")
    print (A, type(A), type(A[0]), type(A[1]))
    #print (config.get_value("ROLE", "sd"), type(config.get_value("ROLE", "sd")))
    #print (config.get_value("ROLE", "sd", dtype="int"), type(config.get_value("ROLE", "sd", dtype="int")))
    #print (config.get_value("UPDATE", "new_key", dtype="int"), type(config.get_value("ROLE", "sd", dtype="int")))
