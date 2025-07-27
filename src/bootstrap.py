import os

import easyocr
import streamlit as st
from openai import AzureOpenAI


@st.cache_resource
def load_resources():
    reader = easyocr.Reader(["en"], verbose=False)
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://aimicroservices-openai-fr.openai.azure.com/",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    return reader, client
