"""
Entry point script for the SCM Analysis Demo.
"""
import os
import sys
import shutil
from src.workflow import run_agent_workflow, run_scm_workflow

# Only import streamlit if needed for web interface
try:
    import streamlit as st
except ImportError:
    st = None

def remove_artifact_folder(folder_path="./artifacts/"):
    """
    ./artifact/ 폴더가 존재하면 삭제하는 함수
    
    Args:
        folder_path (str): 삭제할 폴더 경로
    """
    if os.path.exists(folder_path):
        print(f"'{folder_path}' 폴더를 삭제합니다...")
        try:
            # 폴더와 그 내용을 모두 삭제
            shutil.rmtree(folder_path)
            print(f"'{folder_path}' 폴더가 성공적으로 삭제되었습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
    else:
        print(f"'{folder_path}' 폴더가 존재하지 않습니다.")

def execution(user_query):
    remove_artifact_folder()
    result = run_agent_workflow(
        user_input=user_query,
        debug=False
    )

    # Print the conversation history
    print("\n=== Conversation History ===")
    print ("result", result)
    for history in result["history"]:
        print ("===")
        print (f'agent: {history["agent"]}')
        print (f'message: {history["message"]}')

    return result


def scm_execution(user_query):
    """Execute SCM specialized workflow"""
    print(f"\n🔗 Starting SCM Analysis for: {user_query}")
    
    result = run_scm_workflow(
        user_input=user_query,
        debug=False
    )

    # Print the conversation history
    print("\n=== SCM Analysis History ===")
    print ("result", result)
    for history in result.get("history", []):
        print ("===")
        print (f'agent: {history["agent"]}')
        print (f'message: {history["message"]}')

    return result
    

if __name__ == "__main__":
    
    # Check if --scm flag is provided for SCM workflow
    use_scm = "--scm" in sys.argv
    if use_scm:
        sys.argv.remove("--scm")
    
    if len(sys.argv) > 1: 
        user_query = " ".join(sys.argv[1:])
    else: 
        print("Available modes:")
        print("1. Regular analysis workflow")
        print("2. SCM specialized workflow (add --scm flag)")
        print()
        user_query = input("Enter your query: ")
        
        # Auto-detect SCM queries
        scm_keywords = ["supply chain", "scm", "port", "shipping", "logistics", "disruption", "strike", "transportation"]
        if not use_scm and any(keyword in user_query.lower() for keyword in scm_keywords):
            use_scm_input = input("\nThis appears to be a supply chain query. Use SCM workflow? (y/n): ").lower()
            use_scm = use_scm_input.startswith('y')
    
    # Execute appropriate workflow
    if use_scm:
        print("🔗 Using SCM specialized workflow")
        result = scm_execution(user_query)
    else:
        print("📊 Using regular analysis workflow")
        remove_artifact_folder()
        result = run_agent_workflow(user_input=user_query, debug=False)
        
        # Print the conversation history
        print("\n=== Conversation History ===")
        print ("result", result)
        for history in result["history"]:
            print ("===")
            print (f'agent: {history["agent"]}')
            print (f'message: {history["message"]}')