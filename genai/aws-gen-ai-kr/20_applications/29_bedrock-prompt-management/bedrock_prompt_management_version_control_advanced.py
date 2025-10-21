#!/usr/bin/env python3
"""
Amazon Bedrock Prompt Management Utility - Advanced version
Key features:
- Tag-based Version Management: Create versions with meaningful tags (composite tags)
- Cross-environment Promotion: Automated DEV → PROD promotion process
- Rollback Functionality: Safe rollback to previous versions
- Interactive Interface: User-friendly CLI interface
"""

import boto3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

ENVIRONMENT_CONFIG = {
    'dev': {
        'parameter_store_path': '/prompts/text2sql/dev/current',
        'description': 'Development Environment',
        'default_tags': {
            'Environment': 'DEV',
            'Status': 'TESTING'
        }
    },
    'prod': {
        'parameter_store_path': '/prompts/text2sql/prod/current',
        'description': 'Production Environment',
        'default_tags': {
            'Environment': 'PROD',
            'Status': 'ACTIVE'
        }
    }
}

DEFAULT_REGION = 'us-west-2'
SUPPORTED_ENVIRONMENTS = ['dev', 'prod']

class PromptVersionController:
    def __init__(self, region_name: str = DEFAULT_REGION, environment: str = 'dev'):
        self.bedrock_agent = boto3.client('bedrock-agent', region_name=region_name)
        self.ssm_client = boto3.client('ssm', region_name=region_name)
        self.region = region_name
        self.environment = environment.lower()
        
        # Verify environment settings
        if self.environment not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"Unsupported environment: {environment}. Supported: {SUPPORTED_ENVIRONMENTS}")
        
        self.env_config = ENVIRONMENT_CONFIG[self.environment]
        self.parameter_store_path = self.env_config['parameter_store_path']
        
        print(f"🎯 Initialized for {self.env_config['description']}")
        print(f"📍 Parameter Store: {self.parameter_store_path}")
    
    def get_prompt_id_from_environment(self) -> Optional[str]:
        """
        Retrieve Prompt ID of current environment from Parameter Store
        
        Returns:
            Prompt ID or None
        """
        try:
            response = self.ssm_client.get_parameter(
                Name=self.parameter_store_path,
                WithDecryption=True
            )
            prompt_id = response['Parameter']['Value']
            print(f"✅ Retrieved Prompt ID from {self.environment.upper()}: {prompt_id}")
            return prompt_id
        except ClientError as e:
            print(f"❌ Error retrieving parameter {self.parameter_store_path}: {e}")
            return None
    
    def create_tagged_version(self, prompt_identifier: str, content: str, 
                            environment: str = None, version_tag: str = None, description: str = None) -> Optional[str]:
        """
        Create new version with tags
        
        Args:
            prompt_identifier: Prompt ID
            content: New content
            environment: Environment (default: current environment)
            version_tag: Version tag (e.g., v1.0.0, v1.1.0-beta)
            description: Version description
            
        Returns:
            New version number or None
        """
        # Set environment default value
        if environment is None:
            environment = self.environment
        
        # Set version tag default value
        if version_tag is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M')
            version_tag = f"v1.0.0-{environment}-{timestamp}"
        
        try:
            # 1. Update DRAFT with new content
            current_prompt = self.bedrock_agent.get_prompt(promptIdentifier=prompt_identifier)
            
            # Update variants with new content
            updated_variants = []
            for variant in current_prompt.get('variants', []):
                updated_variant = variant.copy()
                updated_variant['templateConfiguration']['text']['text'] = content
                updated_variants.append(updated_variant)
            
            # Update DRAFT
            self.bedrock_agent.update_prompt(
                promptIdentifier=prompt_identifier,
                name=current_prompt.get('name'),
                description=description or current_prompt.get('description'),
                variants=updated_variants
            )
            
            # 2. Create new version
            version_response = self.bedrock_agent.create_prompt_version(
                promptIdentifier=prompt_identifier,
                description=f"{environment.upper()} {version_tag}: {description or 'Version created'}"
            )
            
            new_version = version_response.get('version')
            new_arn = version_response.get('arn')
            
            # 3. Apply tags (environment + metadata)
            base_tags = ENVIRONMENT_CONFIG.get(environment, {}).get('default_tags', {})
            tags = {
                **base_tags,
                'Version': version_tag,
                'CreatedDate': datetime.now().strftime('%Y-%m-%d'),
                'CreatedTime': datetime.now().strftime('%H:%M:%S'),
                'SourceEnvironment': self.environment.upper()
            }
            
            self.bedrock_agent.tag_resource(
                resourceArn=new_arn,
                tags=tags
            )
            
            print(f"✅ Created version {new_version} with tags:")
            for key, value in tags.items():
                print(f"   {key}: {value}")
            
            return new_version
            
        except ClientError as e:
            print(f"❌ Error creating tagged version: {e}")
            return None
    
    def list_versions_with_tags(self, prompt_identifier: str) -> List[Dict]:
        """
        List all versions of a Prompt with their tags
        
        Args:
            prompt_identifier: Prompt ID
            
        Returns:
            List of version information with tags
        """
        try:
            versions = []
            
            # Get DRAFT version (no tags)
            draft_prompt = self.bedrock_agent.get_prompt(promptIdentifier=prompt_identifier)
            base_arn = draft_prompt.get('arn')
            
            versions.append({
                'version': 'DRAFT',
                'arn': base_arn,
                'name': draft_prompt.get('name'),
                'content': draft_prompt['variants'][0]['templateConfiguration']['text']['text'][:100] + "...",
                'tags': {}  # DRAFT has no tags
            })
            
            # Get numbered versions using ARN format
            version_num = 1
            max_attempts = 20  # Check up to 20 versions
            
            while version_num <= max_attempts:
                try:
                    # Query version using ARN format
                    version_arn = f"{base_arn}:{version_num}"
                    versioned_prompt = self.bedrock_agent.get_prompt(promptIdentifier=version_arn)
                    
                    # Get tags for this version
                    try:
                        tags_response = self.bedrock_agent.list_tags_for_resource(
                            resourceArn=version_arn
                        )
                        tags = tags_response.get('tags', {})
                    except:
                        tags = {}
                    
                    versions.append({
                        'version': str(version_num),
                        'arn': version_arn,
                        'name': versioned_prompt.get('name'),
                        'content': versioned_prompt['variants'][0]['templateConfiguration']['text']['text'][:100] + "...",
                        'tags': tags
                    })
                    
                    version_num += 1
                    
                except ClientError as e:
                    if 'ResourceNotFoundException' in str(e) or 'ValidationException' in str(e):
                        # Skip if version doesn't exist
                        version_num += 1
                        continue
                    else:
                        version_num += 1
                        continue
            
            return versions
            
        except ClientError as e:
            print(f"❌ Error listing versions: {e}")
            return []
    
    def rollback_to_version(self, prompt_identifier: str, target_version: str, 
                          rollback_reason: str = "Manual rollback") -> bool:
        """
        Rollback to a specific version
        
        Args:
            prompt_identifier: Prompt ID
            target_version: Target version number to rollback to
            rollback_reason: Reason for rollback
            
        Returns:
            Success status
        """
        try:
            # 1. Get current DRAFT info (to obtain base ARN)
            current_prompt = self.bedrock_agent.get_prompt(promptIdentifier=prompt_identifier)
            base_arn = current_prompt.get('arn')
            
            # 2. Get target version content
            if target_version == 'DRAFT':
                target_prompt = current_prompt
            else:
                target_arn = f"{base_arn}:{target_version}"
                target_prompt = self.bedrock_agent.get_prompt(promptIdentifier=target_arn)
            
            target_content = target_prompt['variants'][0]['templateConfiguration']['text']['text']
            
            # 3. Update current DRAFT with target version content
            updated_variants = []
            for variant in current_prompt.get('variants', []):
                updated_variant = variant.copy()
                updated_variant['templateConfiguration']['text']['text'] = target_content
                updated_variants.append(updated_variant)
            
            self.bedrock_agent.update_prompt(
                promptIdentifier=prompt_identifier,
                name=current_prompt.get('name'),
                description=f"Rollback to version {target_version}: {rollback_reason}",
                variants=updated_variants
            )
            
            # 4. Create rollback version (for audit trail)
            rollback_version = self.bedrock_agent.create_prompt_version(
                promptIdentifier=prompt_identifier,
                description=f"ROLLBACK to v{target_version} - {rollback_reason}"
            )
            
            # 5. Apply rollback tags
            rollback_arn = rollback_version.get('arn')
            rollback_tags = {
                'Environment': 'ROLLBACK',
                'RollbackFrom': 'DRAFT',
                'RollbackTo': target_version,
                'RollbackDate': datetime.now().strftime('%Y-%m-%d'),
                'RollbackReason': rollback_reason,
                'Status': 'ROLLBACK_COMPLETE',
                'SourceEnvironment': self.environment.upper()
            }
            
            self.bedrock_agent.tag_resource(
                resourceArn=rollback_arn,
                tags=rollback_tags
            )
            
            print(f"✅ Successfully rolled back to version {target_version}")
            print(f"   New rollback version: {rollback_version.get('version')}")
            print(f"   Reason: {rollback_reason}")
            
            return True
            
        except ClientError as e:
            print(f"❌ Error during rollback: {e}")
            return False
    
    def promote_version(self, prompt_identifier: str, from_env: str, to_env: str, 
                       version_tag: str) -> bool:
        """
        Cross-environment version promotion - Update target environment Prompt
        
        Args:
            prompt_identifier: Source environment Prompt ID
            from_env: Source environment
            to_env: Target environment
            version_tag: New version tag
            
        Returns:
            Success status
        """
        try:
            print(f"🔄 Starting promotion from {from_env.upper()} to {to_env.upper()}...")
            
            # 1. Get source environment DRAFT content
            source_prompt = self.bedrock_agent.get_prompt(promptIdentifier=prompt_identifier)
            source_content = source_prompt['variants'][0]['templateConfiguration']['text']['text']
            
            print(f"📋 Source content: {source_content[:100]}...")
            
            # 2. Get target environment Prompt ID from Parameter Store
            target_param_path = ENVIRONMENT_CONFIG[to_env]['parameter_store_path']
            
            try:
                target_response = self.ssm_client.get_parameter(
                    Name=target_param_path,
                    WithDecryption=True
                )
                target_prompt_id = target_response['Parameter']['Value']
                print(f"🎯 Target Prompt ID ({to_env.upper()}): {target_prompt_id}")
            except ClientError as e:
                print(f"❌ Could not get target environment Prompt ID: {e}")
                return False
            
            # 3. Get target environment current Prompt info
            try:
                target_prompt = self.bedrock_agent.get_prompt(promptIdentifier=target_prompt_id)
                print(f"📋 Current target content: {target_prompt['variants'][0]['templateConfiguration']['text']['text'][:100]}...")
            except ClientError as e:
                print(f"❌ Could not get target prompt details: {e}")
                return False
            
            # 4. Update target environment DRAFT with source content
            updated_variants = []
            for variant in target_prompt.get('variants', []):
                updated_variant = variant.copy()
                updated_variant['templateConfiguration']['text']['text'] = source_content
                updated_variants.append(updated_variant)
            
            self.bedrock_agent.update_prompt(
                promptIdentifier=target_prompt_id,
                name=target_prompt.get('name'),
                description=f"Promoted from {from_env.upper()} - {version_tag}",
                variants=updated_variants
            )
            
            print(f"✅ Updated {to_env.upper()} DRAFT with {from_env.upper()} content")
            
            # 5. Create new version in target environment
            version_response = self.bedrock_agent.create_prompt_version(
                promptIdentifier=target_prompt_id,
                description=f"Promoted from {from_env.upper()} to {to_env.upper()} - {version_tag}"
            )
            
            new_version = version_response.get('version')
            new_arn = version_response.get('arn')
            
            # 6. Apply promotion tags
            base_tags = ENVIRONMENT_CONFIG.get(to_env, {}).get('default_tags', {})
            promotion_tags = {
                **base_tags,
                'Version': version_tag,
                'PromotedFrom': from_env.upper(),
                'PromotedDate': datetime.now().strftime('%Y-%m-%d'),
                'PromotedTime': datetime.now().strftime('%H:%M:%S'),
                'SourcePromptId': prompt_identifier,
                'PromotionType': 'ENVIRONMENT_PROMOTION'
            }
            
            self.bedrock_agent.tag_resource(
                resourceArn=new_arn,
                tags=promotion_tags
            )
            
            print(f"✅ Successfully promoted from {from_env.upper()} to {to_env.upper()}")
            print(f"   Source Prompt ID: {prompt_identifier}")
            print(f"   Target Prompt ID: {target_prompt_id}")
            print(f"   New version in {to_env.upper()}: {new_version} ({version_tag})")
            print(f"   Applied tags: {promotion_tags}")
            
            # 7. Post-promotion verification
            verification_prompt = self.bedrock_agent.get_prompt(promptIdentifier=target_prompt_id)
            verification_content = verification_prompt['variants'][0]['templateConfiguration']['text']['text']
            
            if verification_content == source_content:
                print(f"✅ Verification successful: Content matches in {to_env.upper()}")
                return True
            else:
                print(f"⚠️ Verification warning: Content may not match exactly")
                return True
            
        except Exception as e:
            print(f"❌ Error during promotion: {e}")
            import traceback
            traceback.print_exc()
            return False

def interactive_demo():
    """Run interactive demo"""
    print("🌍 Environment Selection")
    print("=" * 40)
    print("Available environments:")
    for env in SUPPORTED_ENVIRONMENTS:
        config = ENVIRONMENT_CONFIG[env]
        print(f"  {env.upper()}: {config['description']}")
        print(f"    Parameter Store: {config['parameter_store_path']}")
    
    # Environment selection
    while True:
        selected_env = input(f"\n👉 Select environment ({'/'.join(SUPPORTED_ENVIRONMENTS)}): ").lower().strip()
        if selected_env in SUPPORTED_ENVIRONMENTS:
            break
        print(f"❌ Invalid environment. Please choose from: {', '.join(SUPPORTED_ENVIRONMENTS)}")
    
    # Initialize controller with selected environment
    controller = PromptVersionController(environment=selected_env)
    
    # Get Prompt ID from environment
    prompt_id = controller.get_prompt_id_from_environment()
    if not prompt_id:
        print("❌ Could not retrieve prompt ID from Parameter Store")
        manual_input = input("Enter Prompt ID manually (or press Enter to exit): ").strip()
        if not manual_input:
            return
        prompt_id = manual_input
    
    print(f"\n🎯 Using Prompt ID: {prompt_id}")
    print(f"🌍 Working in {selected_env.upper()} environment")
    
    while True:
        print("\n" + "="*60)
        print(f"🏷️  Bedrock Prompt Version Control & Rollback Demo ({selected_env.upper()})")
        print("="*60)
        print("1. 📋 List all versions with tags")
        print("2. 🏷️  Create new tagged version")
        print("3. 🔄 Rollback to specific version")
        print("4. 🚀 Promote version between environments")
        print("5. 🔄 Switch environment")
        print("6. 🚪 Exit")
        
        choice = input("\n👉 Select option (1-6): ")
        
        if choice == "1":
            print(f"\n📋 Listing all versions for {selected_env.upper()}...")
            versions = controller.list_versions_with_tags(prompt_id)
            
            print(f"\n📊 Found {len(versions)} versions:")
            for version_info in versions:
                print(f"\n🔖 Version: {version_info['version']}")
                print(f"   Content: {version_info['content']}")
                if version_info['tags']:
                    env = version_info['tags'].get('Environment', 'N/A')
                    ver = version_info['tags'].get('Version', 'N/A')
                    status = version_info['tags'].get('Status', 'N/A')
                    source = version_info['tags'].get('SourceEnvironment', 'N/A')
                    print(f"   🏷️  {env} | {ver} | {status} | Source: {source}")
                else:
                    print('   🏷️  DRAFT | No tags')
        
        elif choice == "2":
            print(f"\n🏷️ Creating new tagged version in {selected_env.upper()}...")
            content = input("Enter new content: ")
            
            # Provide environment-specific default
            default_version = f"v1.0.0-{selected_env}"
            version_tag = input(f"Enter version tag (default: {default_version}): ").strip()
            if not version_tag:
                version_tag = default_version
                
            description = input("Enter description (optional): ")
            
            new_version = controller.create_tagged_version(
                prompt_id, content, version_tag=version_tag, description=description
            )
            
            if new_version:
                print(f"✅ Created version {new_version} successfully!")
        
        elif choice == "3":
            print(f"\n🔄 Rolling back in {selected_env.upper()} environment...")
            
            # Show version list first
            versions = controller.list_versions_with_tags(prompt_id)
            print("\nAvailable versions:")
            for i, version_info in enumerate(versions):
                env_tag = version_info['tags'].get('Environment', 'N/A')
                ver_tag = version_info['tags'].get('Version', 'N/A')
                print(f"  {i+1}. Version {version_info['version']} - {env_tag} {ver_tag}")
            
            target_version = input("\nEnter version number to rollback to: ")
            reason = input("Enter rollback reason: ")
            
            success = controller.rollback_to_version(prompt_id, target_version, reason)
            if success:
                print("✅ Rollback completed successfully!")
        
        elif choice == "4":
            print(f"\n🚀 Promoting version from {selected_env.upper()}...")
            
            # Target environment selection
            other_envs = [env for env in SUPPORTED_ENVIRONMENTS if env != selected_env]
            print(f"Available target environments: {', '.join(other_envs)}")
            
            to_env = input(f"To environment ({'/'.join(other_envs)}): ").lower().strip()
            if to_env not in other_envs:
                print("❌ Invalid target environment")
                continue
                
            version_tag = input("New version tag (e.g., v1.3.0): ")
            
            success = controller.promote_version(prompt_id, selected_env, to_env, version_tag)
            if success:
                print("✅ Promotion completed successfully!")
        
        elif choice == "5":
            print("\n🔄 Switching environment...")
            # Recursive call for environment re-selection
            interactive_demo()
            return
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("⚠️ Invalid option, please try again")

def main():
    """Main execution function"""
    print("🚀 Starting Prompt Version Control")
    print("This demo will show you how to:")
    print("  • Select working environment (DEV/PROD)")
    print("  • Create tagged versions")
    print("  • List versions with tags")
    print("  • Rollback to previous versions")
    print("  • Promote between environments")
    
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()