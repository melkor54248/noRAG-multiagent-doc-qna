import streamlit as st
from openai import AzureOpenAI
from configuration.config import ConfigLoader
import ui

# Initialize configuration
if 'config' not in st.session_state:
    st.session_state.config = ConfigLoader()

# Configure OpenAI
azure_config = st.session_state.config.get_azure_config()
client = AzureOpenAI(
    api_key=azure_config['api_key'],
    api_version=azure_config['api_version'],
    azure_endpoint=azure_config['azure_endpoint']
)
deployment_name = azure_config['deployment_name']

ui.main_ui(deployment_name, client)
