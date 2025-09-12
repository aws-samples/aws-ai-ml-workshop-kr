import json
import boto3
import botocore
from botocore.exceptions import ClientError

class iam_handler():
    
    def __init__(self, ):
        
        self.client = boto3.client('iam')
        self.dicTrustRelationship = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "XXX"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
    
    def _has_role(self, strRoleName):
        
        bResponse = False
        if strRoleName in set(self.get_all_role_names()): bResponse=True
        
        return bResponse

    def _generate_trust_relationship(self, listService):
        
        self.dicTrustRelationship["Statement"][0]["Principal"]["Service"] = listService
        
        return json.dumps(self.dicTrustRelationship)
    
    def attach_policy(self, strRoleName, strPolicyArn):
        
        response = self.client.attach_role_policy(
            RoleName=strRoleName,
            PolicyArn=strPolicyArn
        )
    
    def _detach_policy(self, strRoleName, strPolicyArn):
        
        response = self.client.detach_role_policy(
            RoleName=strRoleName,
            PolicyArn=strPolicyArn
        )
        print (response)
        
    def create_role(self, listService, strRoleName, listPolicyArn, strDescription="None"):
        print ("== CREATE ROLE ==")
        if self._has_role(strRoleName):
            print (f"  Role Name: [{strRoleName}] is already exist!!, so, this will be deleted and re-created.")
            self.delete_role(strRoleName)
        
        response = self.client.create_role(
            RoleName=strRoleName,
            AssumeRolePolicyDocument=self._generate_trust_relationship(listService),
            Description=strDescription
        )
        roleArn = response['Role']['Arn']

        for strPolicyArn in listPolicyArn: self.attach_policy(strRoleName, strPolicyArn)
        
        print (f"  Service name: {listService}, \n  Role name: {strRoleName}, \n  Policys: {listPolicyArn}, \n  Role ARN: {roleArn}") 
        print ("== COMPLETED ==")
        return roleArn

    def delete_role(self, strRoleName):
        
        try:
            if self._has_role(strRoleName):
                dicPolicyMap = self.get_policies_for_roles(listRoleNames=[strRoleName])    
                for listPolicy in dicPolicyMap.values():
                    if listPolicy:
                        for dicPolicyInfo in listPolicy:
                            strPolicyArn = dicPolicyInfo['PolicyArn']
                            self._detach_policy(strRoleName, strPolicyArn)
                            print (strPolicyArn)
                        
                response = self.client.delete_role(
                    RoleName=strRoleName
                )
        except ClientError:
            print (f"Couldn't delete role: {strRoleName}.")
            raise

    def get_all_role_names(self, ):
        
        """ Retrieve a list of role names by paginating over list_roles() calls """
        listRoles = []
        role_paginator = self.client.get_paginator('list_roles')
        for response in role_paginator.paginate():
            response_role_names = [r.get('RoleName') for r in response['Roles']]
            listRoles.extend(response_role_names)
        return listRoles

    def get_policies_for_roles(self, listRoleNames):
        """ Create a mapping of role names and any policies they have attached to them by 
            paginating over list_attached_role_policies() calls for each role name. 
            Attached policies will include policy name and ARN.
        """
        dicPolicyMap = {}
        policy_paginator = self.client.get_paginator('list_attached_role_policies')
        for strName in listRoleNames:
            role_policies = []
            for response in policy_paginator.paginate(RoleName=strName):
                role_policies.extend(response.get('AttachedPolicies'))
            dicPolicyMap.update({strName: role_policies})
        return dicPolicyMap
    
    
if __name__ == "__main__":
    
    iam = iam_handler()
   
    lambdaRoleArn = iam.create_role(
        listService=["lambda.amazonaws.com"],
        strRoleName="LambdaTestRole",
        listPolicyArn = [
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        ],
        strDescription="Role for Lambda to call SageMaker functions'")
    print (lambdaRoleArn)
    
    iam.delete_role("LambdaTestRole")