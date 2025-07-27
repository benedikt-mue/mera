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

default_prompt = """I extracted this text from my receipt, can you please find an appropriate category to classify the expense.
The categories are: 'Food', 'Transport', 'Entertainment', 'Shopping', 'Health', 'Other'.
If the text does not match any of the categories, respond with 'Other'. If you are unsure, also respond with 'Other'.
Please follow these instructions carefully. Do not include any other text or explanations.
Output 'Category', 'Date', 'Company or Point of Sale', 'Location', 'Currency', 'Amount Paid' each into a new line and use these words to map the data exactly as they are.
The amount paid is the total amount on the receipt and usually accompanied by words like 'Total', 'Paid', or 'Amount' and a currency indicator.
If the information is not available, put 'na' instead. Format the date to yyyy-mm-dd. Translate all outputted words into english."""

custom_prompt = st.text_area(
    "🛠️ Customize extraction prompt", value=default_prompt, height=200
)

# custom_prompt += f"The output is in a list format, each item represents a bounding box, the text detected and confident level, respectively: "
custom_prompt += "The text from the receipts is extracted from left top to bottom right in a list format, which means "
"that the first words in the list are at the top left of a receipt and the last words of the list are on the bottom right. Here is the list: "


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

# Store uploaded files - Add only new files (avoid duplicates by name)
if uploaded:
    existing_filenames = {f.name for f in st.session_state.uploaded_files}
    new_files = [f for f in uploaded if f.name not in existing_filenames]

    if new_files:
        st.session_state.uploaded_files.extend(new_files)
        logger.info(f"Added new files: {[f.name for f in new_files]}")

        # Force uploader to reset (so duplicate files disappear from UI)
        current_index = int(st.session_state.uploader_key.split("_")[1])
        st.session_state.uploader_key = f"uploader_{current_index + 1}"
        st.rerun()
    else:
        st.warning("⚠️ All selected files were already uploaded.")


# Process uploaded images (only unprocessed ones)
if st.session_state.uploaded_files:
    for uploaded_file in st.session_state.uploaded_files:
        file_id = uploaded_file.name

        # st.subheader(f"🧾 File: `{file_id}`")

        # If already processed, show stored results
        reanalyze_key = f"reanalyze_{file_id}"

        # Check if file already processed
        already_processed = (
            file_id in st.session_state.ocr_results
            and file_id in st.session_state.llm_outputs
        )

        col1, col2 = st.columns([6, 1])  # Adjust ratio as needed

        st.markdown("---")
        st.subheader(f"🧾 File: `{file_id}`")

        reanalyze_triggered = st.button("🔁", key=reanalyze_key)

        # If already processed and no re-analyze clicked
        if already_processed and not reanalyze_triggered:
            st.info("ℹ️ Previously processed. Showing cached result.")
            ocr_text = st.session_state.ocr_results[file_id]
            data_dict = st.session_state.llm_outputs[file_id]
            logger.info(f"Using cached results for {file_id}.")
        else:
            try:
                logger.info(f"Processing file: {file_id}")

                # Open and convert image
                image = Image.open(uploaded_file).convert("RGB")
                logger.info(f"Image {file_id} opened and converted to RGB.")
                image_np = np.array(image)
                logger.info(f"Image {file_id} converted to numpy array.")
                # OCR
                ocr_result = reader.readtext(image_np, detail=0)
                logger.info(f"OCR completed for {file_id}.")
                ocr_text = ocr_result
                full_prompt = f"{custom_prompt} {ocr_result}"

                # Store OCR result
                st.session_state.ocr_results[file_id] = ocr_text

                # LLM extraction
                with st.spinner("🔍 Analyzing with Azure OpenAI..."):
                    logger.debug(f"Sending OCR text to LLM for {file_id}.")
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that helps classify expenses from receipts.",
                            },
                            {"role": "user", "content": full_prompt},
                        ],
                        max_tokens=4096,
                        temperature=1,
                        top_p=1.0,
                        model="gpt-4-32k-0613",
                    )
                    raw_output = response.choices[0].message.content
                    logger.info(f"LLM response for {file_id}: {raw_output}")

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

                    # Store LLM result
                    st.session_state.llm_outputs[file_id] = data_dict

            except Exception as e:
                logger.error(f"Error processing {file_id}: {e}", exc_info=True)
                st.error(f"❌ Error processing `{file_id}`: {e}")
                continue  # skip display

        # Always show OCR + results
        with st.expander("📝 Show extracted OCR text"):
            st.code(ocr_text)

        st.markdown("#### 📄 Extracted Information")
        st.table(pd.DataFrame.from_dict(data_dict, orient="index", columns=["Value"]))

        # Always show image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"🖼️ {file_id}", use_container_width=True)
        logger.info(f"Image displayed for {file_id}.")
