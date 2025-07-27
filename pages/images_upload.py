import io
import logging
import os
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener

from src.auth import load_authenticator
from src.bootstrap import load_resources

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# Authenticate user
authenticator = load_authenticator()
try:
    auth = authenticator.login("main")
    logger.info("User authentication attempted.")
except Exception as e:
    logger.error(f"Authentication error: {e}")
    st.error(e)

if not st.session_state.get("authentication_status"):
    logger.warning("Authentication failed or not present in session state.")
    st.stop()

# Register HEIF opener
register_heif_opener()
logger.info("HEIF opener registered.")

reader, client = load_resources()
logger.info("Resources loaded (OCR reader and LLM client).")
st.title("Upload and Analyze Receipt")

st.markdown(" ")  # One empty line

# Prompt for LLM
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
end_custom_prompt = (
    custom_prompt
    + "The text from the receipts is extracted from left top to bottom right in a list format. Here is the list: "
)

# st.markdown(" ")  # One empty line
st.markdown("----")  # One empty line


allowed_fields = [
    "category",
    "date",
    "company or point of sale",
    "location",
    "currency",
    "amount paid",
]

st.markdown("### 🧩 Filename Pattern Builder")

# Multi-select fields
selected_fields = st.multiselect(
    "Select fields to include in filename (order matters):",
    options=allowed_fields,
    default=["date", "category", "amount paid"],
)

# Separator choice
separator = st.radio("Choose a separator:", options=["_", "-"], horizontal=True)

# Preview & internal pattern
if selected_fields:
    rename_pattern = separator.join(selected_fields)
    st.markdown(f"**🔤 Preview pattern:** `{rename_pattern}`")
else:
    rename_pattern = ""
    st.warning("Please select at least one field.")

# Store in session_state for reuse
st.session_state.rename_pattern = rename_pattern


# Initialize session state for uploaded files and results
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    logger.info("Session state: initialized uploaded_files.")
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"
    logger.info("Session state: initialized uploader_key.")
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}
    logger.info("Session state: initialized ocr_results.")
if "llm_outputs" not in st.session_state:
    st.session_state.llm_outputs = {}
    logger.info("Session state: initialized llm_outputs.")
if "saved_images" not in st.session_state:
    st.session_state.saved_images = []
    logger.info("Session state: initialized saved_images.")

st.markdown("----")  # One empty line
st.markdown(" ")  # One empty line

# File uploader
uploaded = st.file_uploader(
    "📷 Upload one or more receipt images...",
    type=["jpg", "jpeg", "png", "heic"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key,
)

st.markdown("")  # One empty line

col1, col2 = st.columns([9, 1])
with col1:
    if st.session_state.uploaded_files:
        if st.button("🗑️ Remove Uploaded Images"):
            logger.info("User triggered removal of uploaded images.")
            st.session_state.uploaded_files = []
            st.session_state.ocr_results = {}
            st.session_state.llm_outputs = {}
            st.session_state.saved_images = []
            current_index = int(st.session_state.uploader_key.split("_")[1])
            st.session_state.uploader_key = f"uploader_{current_index + 1}"
            st.rerun()
with col2:
    # batch download of all saved images
    if st.session_state.saved_images:
        logger.info(
            f"Preparing ZIP for {len(st.session_state.saved_images)} saved images."
        )
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for fname, fbytes in st.session_state.saved_images:
                zip_file.writestr(fname, fbytes)
        zip_buffer.seek(0)
        st.download_button(
            label="📁 Download All",
            data=zip_buffer,
            file_name=f"renamed_receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            key="batch_zip_download",
        )

st.markdown(" ")  # One empty line

# Deduplication
if uploaded:
    logger.info(f"{len(uploaded)} files uploaded by user.")
    existing_filenames = {f.name for f in st.session_state.uploaded_files}
    new_files = [f for f in uploaded if f.name not in existing_filenames]
    if new_files:
        logger.info(f"{len(new_files)} new files to add: {[f.name for f in new_files]}")
        st.session_state.uploaded_files.extend(new_files)
        current_index = int(st.session_state.uploader_key.split("_")[1])
        st.session_state.uploader_key = f"uploader_{current_index + 1}"
        st.rerun()
    else:
        logger.warning("All selected files were already uploaded.")
        st.warning("⚠️ All selected files were already uploaded.")

st.markdown("---")  # Horizontal rule

# Process each uploaded file
for uploaded_file in st.session_state.uploaded_files:
    file_id = uploaded_file.name
    reanalyze_key = f"reanalyze_{file_id}"
    save_key = f"save_{file_id}"
    already_processed = (
        file_id in st.session_state.ocr_results
        and file_id in st.session_state.llm_outputs
    )
    st.subheader(f"📄 File: `{file_id}`")

    col1, col2 = st.columns([10.5, 1])
    with col1:
        reanalyze_triggered = st.button("🔁 Rerun", key=reanalyze_key)
    with col2:
        if (
            file_id in st.session_state.llm_outputs
            and file_id in st.session_state.ocr_results
        ):
            data_dict = st.session_state.llm_outputs[file_id]

            new_filename = (
                separator.join(selected_fields) + os.path.splitext(file_id)[1]
            )

            image = Image.open(uploaded_file).convert("RGB")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format=image.format or "JPEG")
            image_bytes.seek(0)
            image_data = image_bytes.read()

            if new_filename not in [f[0] for f in st.session_state.saved_images]:
                st.session_state.saved_images.append((new_filename, image_data))
                logger.info(f"Saved image as {new_filename}.")

            mime_type = (
                "image/jpeg"
                if new_filename.lower().endswith((".jpg", ".jpeg"))
                else "image/png"
            )

            st.download_button(
                label="⬇️ Download",
                data=image_data,
                file_name=new_filename,
                mime=mime_type,
                key=f"download_inline_{file_id}_{datetime.now().timestamp()}",
            )

    if already_processed and not reanalyze_triggered:
        logger.info(f"Using cached OCR/LLM results for {file_id}.")
        ocr_text = st.session_state.ocr_results[file_id]
        data_dict = st.session_state.llm_outputs[file_id]
    else:
        try:
            logger.info(f"Processing file {file_id} with OCR.")
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            ocr_result = reader.readtext(image_np, detail=0)
            ocr_text = ocr_result
            st.session_state.ocr_results[file_id] = ocr_text

            full_prompt = f"{end_custom_prompt} {ocr_result}"
            logger.info(f"Sending prompt to LLM for {file_id}.")
            with st.spinner("🔍 Analyzing with Azure OpenAI..."):
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
                expected_keys = [
                    "category",
                    "date",
                    "company or point of sale",
                    "location",
                    "currency",
                    "amount paid",
                ]
                data_dict = {key: "na" for key in expected_keys}
                for line in raw_output.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        for expected in expected_keys:
                            if key == expected.lower():
                                data_dict[expected] = value
                st.session_state.llm_outputs[file_id] = data_dict
                logger.info(f"LLM output processed for {file_id}.")
        except Exception as e:
            logger.error(f"Error processing `{file_id}`: {e}")
            st.error(f"Error processing `{file_id}`: {e}")
            continue

    # Show extracted OCR results
    editable_df = (
        pd.DataFrame.from_dict(data_dict, orient="index", columns=["Value"])
        .rename_axis("Field")
        .reset_index()
    )
    with st.expander("📝 Show Results", expanded=True):
        edited_df = st.data_editor(
            editable_df,
            num_rows="fixed",
            use_container_width=True,
            key=f"editor_{file_id}",
        )

        # Show image
        with st.expander("🖼️ Show Picture", expanded=True):
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"🖼️ {file_id}", use_container_width=True)
            with st.expander("🖋️ Show Raw Extracted Info"):
                st.code(ocr_text)
