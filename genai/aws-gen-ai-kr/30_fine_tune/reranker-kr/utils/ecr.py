import os

class ecr_handler():
    
    def __init__(self, ):
        
        self.strUriSuffix = "amazonaws.com"
        
    def build_docker(self, strDockerDir, strDockerFile, strRepositoryName, strRegionName, strAccountId):
        
        strCurrentWD = os.getcwd()
        print (os.getcwd())
        
        os.chdir(strDockerDir)
        print (os.getcwd())
        print ("strDockerFile", strDockerFile)
        
        strQuery = "".join(["aws ecr get-login --region ", "'", strRegionName, "' ", "--registry-ids ", "'", strAccountId, "' ", "--no-include-email"])
        
        print (strQuery)
        strResponse = os.popen(strQuery).read()
        strResponse = os.popen(strResponse).read()
        print (strResponse)
        
        
        #strQuery = "".join(["docker build -t ", "'", strRepositoryName, "' ", "."])
        strQuery = "".join(["docker build -f ", "'", strDockerFile, "' ", "-t ", "'", strRepositoryName, "' ", "."])
        strResponse = os.popen(strQuery).read()
        
        print (strResponse)
        
        os.chdir(strCurrentWD)
        print (os.getcwd())

    def register_image_to_ecr(self, strRegionName, strAccountId, strRepositoryName, strTag):
        
        if not strTag.startswith(":"): strTag = ":" + strTag
        
        print ("== REGISTER AN IMAGE TO ECR ==")
        processing_repository_uri = "{}.dkr.ecr.{}.{}/{}".format(strAccountId, strRegionName, self.strUriSuffix, strRepositoryName + strTag)
        print (f'  processing_repository_uri: {processing_repository_uri}')
        
        strQuery = "".join(["aws ecr get-login --region ", "'", strRegionName, "' ", "--registry-ids ", "'", strAccountId, "' ", "--no-include-email"])
        
        print (strQuery)
        strResponse = os.popen(strQuery).read()
        strResponse = os.popen(strResponse).read()
        print (strResponse)
        
        strQuery = "".join(["aws ecr create-repository --repository-name ", "'", strRepositoryName, "'"])
        print (strQuery)
        strResponse = os.popen(strQuery).read()
        
        strImageTag = strRepositoryName + strTag
        strQuery = "".join(["docker tag ", "'", strImageTag, "' ",  "'", processing_repository_uri, "'"])
        print (strQuery)
        strResponse = os.popen(strQuery).read()
        
        strQuery = "".join(["docker push ", "'", processing_repository_uri, "'"])
        print (strQuery)
        strResponse = os.popen(strQuery).read()
        
        print ("== REGISTER AN IMAGE TO ECR ==")
        print ("==============================")
        
        return processing_repository_uri
    

if __name__ == "__main__":
    
    iam = iam_handler()