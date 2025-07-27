import io
import logging

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener

from src.auth import load_authenticator
from src.bootstrap import load_resources

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)
logger = logging.getLogger(__name__)

authenticator = load_authenticator()

# try:
#     auth = authenticator.login("main")
# except Exception as e:
#     st.error(e)

if not st.session_state["authentication_status"]:
    st.stop()

logger.debug(
    "User authentication status: %s", st.session_state["authentication_status"]
)

register_heif_opener()
logger.info("HEIF opener registered.")

reader, client = load_resources()
logger.info("Resources loaded: OCR reader and OpenAI client.")

st.title("Upload and Analyze Receipt")

default_prompt = """
    I extracted this text from my receipt, can you please find an appropriate category to classify the expense. 
    The categories are: 'Food', 'Transport', 'Entertainment', 'Shopping', 'Health', 'Other'. 
    If the text does not match any of the categories, respond with 'Other'. If you are unsure, also respond with 'Other'. 
    Please follow these instructions carefully. Do not include any other text or explanations. 
    Output 'Category', 'Date', 'Company or Point of Sale', 'Location', 'Currency', 'Amount Paid' each into a new line and use these words to map the data exactly as they are.
    If the information is not available, put 'na' instead. Format the date to yyyy-mm-dd. Translate all outputted words into english.
    """

custom_prompt = st.text_area(
    "🛠️ Customize extraction prompt", value=default_prompt, height=200
)

# custom_prompt += f"The output is in a list format, each item represents a bounding box, the text detected and confident level, respectively: "
custom_prompt += "The text from the receipts is extracted from left top to bottom right and put chronologically in a list format, here is the list: "


# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "llm_outputs" not in st.session_state:
    st.session_state.llm_outputs = {}

# Upload files (key is dynamic and resets on remove)
uploaded = st.file_uploader(
    "📷 Upload one or more receipt images...",
    type=["jpg", "jpeg", "png", "heic"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key,
)

# Remove button – visible only if files are uploaded
if st.session_state.uploaded_files:
    if st.button("🗑️ Remove Uploaded Images"):
        st.session_state.uploaded_files = []
        st.session_state.ocr_results = {}
        st.session_state.llm_outputs = {}
        current_index = int(st.session_state.uploader_key.split("_")[1])
        st.session_state.uploader_key = f"uploader_{current_index + 1}"
        st.rerun()

# Store uploaded files
if uploaded:
    st.session_state.uploaded_files = uploaded

# Process uploaded images
if st.session_state.uploaded_files:
    for uploaded_file in st.session_state.uploaded_files:
        st.markdown("---")
        st.subheader(f"🧾 File: `{uploaded_file.name}`")
        logger.info(f"Processing file: {uploaded_file.name}")

        try:
            # Open and convert image
            image = Image.open(uploaded_file).convert("RGB")
            logger.info(f"Image {uploaded_file.name} opened and converted to RGB.")
            image_np = np.array(image)

            # OCR
            ocr_result = reader.readtext(image_np, detail=0)
            logger.info(f"OCR completed for {uploaded_file.name}.")
            ocr_text = ocr_result
            full_prompt = f"{custom_prompt} {ocr_result}"  # "\n".join(ocr_result)

            with st.expander("📝 Show extracted OCR text"):
                st.code(ocr_text)

            # LLM extraction
            with st.spinner("🔍 Analyzing with Azure OpenAI..."):
                logger.debug(f"Sending OCR text to LLM for {uploaded_file.name}.")
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": full_prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.9,
                    top_p=1.0,
                    model="gpt-4-32k-0613",
                )
                raw_output = response.choices[0].message.content
                logger.info(f"LLM response for {uploaded_file.name}: {raw_output}")

                # Robust dict parsing
                expected_keys = [
                    "category",
                    "date",
                    "company or point of sale",
                    "location",
                    "currency",
                    "amount paid",
                ]
                data_dict = {key: "n/a" for key in expected_keys}
                for line in raw_output.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        for expected in expected_keys:
                            if key == expected.lower():
                                data_dict[expected] = value

                st.markdown("#### 📄 Extracted Information")
                st.table(
                    pd.DataFrame.from_dict(data_dict, orient="index", columns=["Value"])
                )

            st.image(image, caption=f"🖼️ {uploaded_file.name}", use_container_width=True)
            logger.info(f"Image displayed for {uploaded_file.name}.")

        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}", exc_info=True)
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
