import streamlit as st
import os
import sys
import shutil
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가 (필요한 경우)
sys.path.append('..')

# Bedrock-Manus framework 가져오기
try:
    from src.workflow import run_agent_workflow
except ImportError:
    st.error("Bedrock-Manus 프레임워크를 불러올 수 없습니다. 코드가 올바른 위치에 있는지 확인하세요.")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="Bedrock-Manus AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 정의
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34A853;
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: #f7f7f7;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .message-user {
        background-color: #E8F0FE;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .message-agent {
        background-color: #F4F4F4;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .agent-name {
        font-weight: bold;
        color: #4285F4;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """헤더 섹션을 표시합니다."""
    st.markdown('<div class="main-header">Bedrock-Manus AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Amazon Bedrock 기반 AI 자동화 프레임워크</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Bedrock-Manus**는 Amazon Bedrock 서비스를 활용한 AI 자동화 프레임워크입니다.
    비즈니스 사용 사례에 맞게 최적화된 이 프레임워크는 복잡한 작업을 다양한 AI 에이전트가 
    협력하여 처리할 수 있도록 설계되었습니다.
    """)

def remove_artifact_folder(folder_path="./artifacts/"):
    """
    ./artifact/ 폴더가 존재하면 삭제하는 함수
    
    Args:
        folder_path (str): 삭제할 폴더 경로
    """
    if os.path.exists(folder_path):
        try:
            # 폴더와 그 내용을 모두 삭제
            shutil.rmtree(folder_path)
            st.sidebar.success(f"'{folder_path}' 폴더가 성공적으로 삭제되었습니다.")
            # artifacts 폴더 재생성
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            st.sidebar.error(f"오류 발생: {e}")
    else:
        st.sidebar.warning(f"'{folder_path}' 폴더가 존재하지 않습니다.")
        # artifacts 폴더 생성
        os.makedirs(folder_path, exist_ok=True)

def display_conversation_history(history):
    """대화 기록을 표시하는 함수"""
    if not history:
        return
    
    with st.expander("대화 기록 보기", expanded=False):
        for msg in history:
            agent = msg.get("agent", "시스템")
            message = msg.get("message", "")
            
            st.markdown(f"<div class='message-agent'><span class='agent-name'>{agent}</span>: {message}</div>", 
                       unsafe_allow_html=True)

def display_artifacts():
    """생성된 artifacts를 표시하는 함수"""
    artifacts_path = Path("./artifacts/")
    if not artifacts_path.exists():
        return
    
    artifacts = list(artifacts_path.glob("**/*"))
    
    if not artifacts:
        return
    
    with st.expander("생성된 아티팩트 파일", expanded=True):
        st.write("다음 파일들이 생성되었습니다:")
        
        for file_path in artifacts:
            if file_path.is_file():
                rel_path = file_path.relative_to(".")
                st.download_button(
                    label=f"📄 {file_path.name} 다운로드",
                    data=open(file_path, "rb").read(),
                    file_name=file_path.name,
                    mime="application/octet-stream",
                    key=f"download_{str(rel_path).replace('/', '_')}"
                )
                
                # 이미지 파일일 경우 미리보기 표시
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                    st.image(str(file_path), caption=file_path.name)
                
                # 텍스트 파일일 경우 내용 표시
                elif file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css']:
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            content = f.read()
                            with st.expander(f"{file_path.name} 내용 보기"):
                                st.code(content, language=file_path.suffix.lower()[1:])
                        except UnicodeDecodeError:
                            st.warning(f"{file_path.name} 파일은 텍스트 형식이 아닙니다.")

def main():
    """메인 애플리케이션 함수"""
    display_header()
    
    # 사이드바 설정
    st.sidebar.title("설정")
    
    # Debug 모드 설정
    debug_mode = st.sidebar.checkbox("Debug 모드", value=False)
    
    # Artifacts 폴더 정리 버튼
    if st.sidebar.button("Artifacts 폴더 정리"):
        remove_artifact_folder()
        st.experimental_rerun()
    
    # 사용자 쿼리 입력
    st.subheader("질문 또는 작업을 입력하세요")
    user_query = st.text_area("", height=100, placeholder="예: COVID-19 백신의 효과에 대한 최신 연구 요약을 만들어주세요.")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("실행", type="primary")
    
    # 세션 상태 초기화
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # 결과 처리
    if submit_button and user_query:
        with st.spinner("Bedrock-Manus AI가 작업을 처리 중입니다..."):
            try:
                # 진행 상태 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 진행 상태 업데이트를 위한 더미 루프 (실제 진행률을 측정할 수 없을 때 사용)
                for i in range(100):
                    # 실제 작업은 마지막에 수행
                    if i < 99:
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                        status_text.text(f"처리 중... {i+1}%")
                    else:
                        # 실제 워크플로우 실행
                        result = run_agent_workflow(user_input=user_query, debug=debug_mode)
                        progress_bar.progress(100)
                        status_text.text("완료!")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
                
                # 결과 표시
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.subheader("결과")
                
                # 세션 기록 업데이트
                st.session_state.history = result.get("history", [])
                
                # 최종 결과 메시지 표시 (마지막 메시지)
                if st.session_state.history:
                    last_message = st.session_state.history[-1].get("message", "결과가 없습니다.")
                    st.markdown(f"<div class='message-agent'>{last_message}</div>", unsafe_allow_html=True)
                else:
                    st.info("결과가 없습니다.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 대화 기록 표시
                display_conversation_history(st.session_state.history)
                
                # 생성된 아티팩트 표시
                display_artifacts()
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
    
    # 이전 대화 기록이 있으면 표시
    elif st.session_state.history:
        display_conversation_history(st.session_state.history)
        display_artifacts()
    
    # 푸터
    st.markdown("---")
    st.markdown("© 2025 Bedrock-Manus | Amazon Bedrock 기반 AI 자동화 프레임워크")

if __name__ == "__main__":
    main()
