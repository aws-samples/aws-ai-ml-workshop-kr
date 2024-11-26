import streamlit as st
import aicc_lib as glib
import json

# Load language data from JSON file
with open('./localization.json', 'r', encoding='utf-8') as f:
    LANG = json.load(f)

# Streamlit page setup
st.set_page_config(page_title=LANG['en']['page_title'], layout="wide")

# Sidebar for language selection
st.sidebar.title("Language")
language = st.sidebar.radio("Select Language", ('English', 'Korean'))
lang = 'en' if language == 'English' else 'ko'

# Set title based on language
st.title(LANG[lang]['main_title'])

# Display the transcript in an expandable area
transcription_file = f"../dataset/aicc_transcription_{lang}.txt"
with open(transcription_file, 'r', encoding='utf-8') as file:
    transcription_text = file.read()

with st.expander(LANG[lang]['view_transcript']):
    st.write(transcription_text)

# Scenario selection
scenario_options = ['consultation_summary', 'consultation_notes', 'email_reply', 'consultation_quality']
scenario_name = st.selectbox(
    LANG[lang]['select_scenario'],
    scenario_options,
    format_func=lambda x: LANG[lang][x]['name']
)
st.write(LANG[lang][scenario_name]['description'])

# Button for generating response and reading prompt to generate response
if st.button(LANG[lang]['generate']):
    response_placeholder = st.empty()
    with open(f"practice/{LANG[lang][scenario_name]['file']}", 'r', encoding='utf-8') as file:
        prompt = file.read()
    glib.get_streaming_response(prompt, transcription_text, response_placeholder, lang)
