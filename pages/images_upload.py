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

# page settings
st.set_page_config(page_title="Upload", layout="wide", page_icon="👻")

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
"""

if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = default_prompt

st.markdown("## 🛠️ Extraction Prompt Customizer")
with st.expander("Show edit prompt", expanded=False):
    if st.button("Reset Prompt"):
        st.session_state.custom_prompt = default_prompt
        st.toast("Prompt reset to default.", icon="🔄")
    new_prompt = st.text_area(
        label="Edit the prompt (changes here do NOT trigger reprocessing)",
        value=st.session_state.custom_prompt,
        height=200,
        key="custom_prompt_editor",
    )
    if new_prompt != st.session_state.custom_prompt:
        st.session_state.custom_prompt = new_prompt
        st.toast("Prompt updated. It will be used for the next Analysis.", icon="💡")

end_custom_prompt = (
    (
        st.session_state.custom_prompt
        if st.session_state.custom_prompt
        else default_prompt
    )
    + """Please follow these instructions carefully. Do not include any other text or explanations. Output 'Category', 'Date', 'Company or Point of Sale', 'Location', 'Currency', 'Amount Paid' each into a new line and use these words to map the data exactly as they are.
The amount paid is the total amount on the receipt and usually accompanied by words like 'Total', 'Paid', or 'Amount' and a currency indicator.
If the information is not available, put 'na' instead. Format the date to yyyy-mm-dd. 
Translate all outputted words into english. The text from the receipts is extracted from left top to bottom right in a list format. Here is the list: """
)

st.text_area(
    "🔍 Current Prompt",
    value=end_custom_prompt,
    height=200,
    disabled=True,
    key="current_prompt_display",
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

st.markdown("## 🧩 Filename Pattern Builder")

# Multi-select fields
selected_fields = st.multiselect(
    "Select fields to include in filename (order matters):",
    options=allowed_fields,
    default=["date", "category", "location"],
)

# Separator choice
separator = st.radio("Choose a separator:", options=["_", "-"], horizontal=True)

# Preview & internal pattern
if selected_fields:
    rename_pattern = separator.join(selected_fields)
    st.markdown(f"🔤 **Preview pattern:** `{rename_pattern}`")
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
    st.session_state.uploader_key = 0
    logger.info(f"Session state uploader_key: {st.session_state.uploader_key}.")
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}
    logger.info("Session state: initialized ocr_results.")
if "llm_outputs" not in st.session_state:
    st.session_state.llm_outputs = {}
    logger.info("Session state: initialized llm_outputs.")
if "new_files" not in st.session_state:
    st.session_state.new_files = []
if "saved_images" not in st.session_state:
    st.session_state.saved_images = []
    logger.info("Session state: initialized saved_images.")
if "reanalyze_triggered" not in st.session_state:
    st.session_state.reanalyze_triggered = False
    logger.info("Session state: initialized reanalyze_triggered.")

st.markdown("----")
st.markdown(" ")  # One empty line


def update_key():
    st.session_state.uploader_key += 1


# Upload widget
st.markdown("## 📷 Image Uploader")
uploaded = st.file_uploader(
    "Upload one or more receipt images...",
    type=["jpg", "jpeg", "png", "heic"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

# Process uploads rendering the widget
if uploaded:
    uploaded_names = {f.name for f in uploaded}
    existing_names = {f.name for f in st.session_state.uploaded_files}

    new_files = [f for f in uploaded if f.name not in existing_names]
    duplicate_files = uploaded_names & existing_names  # intersection = duplicates

    if new_files:
        st.session_state.uploaded_files.extend(new_files)
        st.session_state.new_files = new_files

        if duplicate_files:
            st.toast("One or more selected files were already uploaded.", icon="⚠️")
    else:
        st.toast("All selected files were already uploaded.", icon="⚠️")

num_uploaded = len(st.session_state.uploaded_files)
if num_uploaded > 0:
    st.caption(
        f"🧾 {num_uploaded} unique file{'s' if num_uploaded != 1 else ''} uploaded."
    )

# Remove button – visible only if files are uploaded
if st.session_state.uploaded_files:
    if st.button("Remove Uploaded Images", icon="🗑️"):
        logger.info("User triggered removal of uploaded images.")
        st.session_state.uploaded_files = []
        st.session_state.ocr_results = {}
        st.session_state.llm_outputs = {}
        st.session_state.saved_images = []
        st.session_state.new_files = []
        # st.session_state.uploader_key = "uploader_0"
        update_key()
        st.rerun()

st.markdown(
    "### <hr style='margin-bottom: 0; border: 1px solid #ccc;'>",
    unsafe_allow_html=True,
)


# Process each uploaded file
for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
    file_id = uploaded_file.name
    reanalyze_key = f"reanalyze_{file_id}"
    already_processed = (
        file_id in st.session_state.ocr_results
        and file_id in st.session_state.llm_outputs
    )

    if already_processed and not st.session_state.reanalyze_triggered:
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
                    temperature=0.75,
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

    # Compute the new filename from the processed result
    parts = [
        st.session_state.llm_outputs[file_id].get(t, "na") for t in selected_fields
    ]
    new_filename = separator.join(parts) + os.path.splitext(file_id)[1]

    # Prepare renamed file for individual download
    download_buffer = io.BytesIO()
    image = Image.open(uploaded_file)
    logger.info(f"Saving renamed image for {file_id} as {new_filename}.")
    image.save(
        download_buffer, format=image.format or "PNG"
    )  # default to PNG if unknown
    download_buffer.seek(0)

    st.session_state.saved_images.append((new_filename, download_buffer.getvalue()))

    # Batch download
    if (
        idx == 0
        and st.session_state.saved_images
        and len(st.session_state.uploaded_files) > 1
    ):
        logger.info(
            f"Preparing ZIP for {len(st.session_state.uploaded_files)} saved images."
        )
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for fname, fbytes in st.session_state.saved_images:
                zip_file.writestr(fname, fbytes)
        zip_buffer.seek(0)
        st.download_button(
            label=f"Download All Files ({len(st.session_state.uploaded_files)}) ",
            icon="📁",
            data=zip_buffer,
            file_name=f"renamed_receipts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            key="batch_zip_download",
        )

        #     filenames = [fname for fname, _ in st.session_state.saved_images]
        #     if filenames:
        #         st.selectbox(
        #             "📁 Last Saved Filename:",
        #             options=filenames,
        #             index=len(filenames) - 1,
        #             disabled=False,
        #         )
        if st.button(
            f"Rerun All Files ({len(st.session_state.uploaded_files)})",
            icon="🔁",
            key="rerun_all",
        ):
            logger.info("User triggered full rerun via 'Rerun All' button.")
            st.session_state.reanalyze_triggered = True
            st.rerun()
        st.markdown("---")
    st.subheader(f"📄 File: `{file_id}`  ➡️  `{new_filename}`")

    st.download_button(
        label="Download",
        icon="⬇️",
        data=download_buffer,
        file_name=new_filename,
        mime="image/png",
        key=f"download_{file_id}",
    )

    st.session_state.reanalyze_triggered = st.button(
        "Rerun", icon="🔁", key=reanalyze_key
    )
    if st.session_state.reanalyze_triggered:
        logger.info(f"User triggered reanalysis for {file_id}.")
        st.session_state.reanalyze_triggered = True
        # st.session_state.uploader_key = (
        #     f"uploader_{int(st.session_state.uploader_key.split('_')[1]) + 1}"
        # )
        st.rerun()

    # Show extracted OCR results
    editable_df = (
        pd.DataFrame.from_dict(data_dict, orient="index", columns=["Value"])
        .rename_axis("Field")
        .reset_index()
    )
    with st.expander("Show Results", icon="📝", expanded=True):
        edited_df = st.data_editor(
            editable_df,
            num_rows="fixed",
            use_container_width=True,
            key=f"editor_{file_id}",
        )

        # Show image
        with st.expander("Show Picture", icon="🖼️", expanded=False):
            st.image(image, caption=f"🖼️ {file_id}", use_container_width=True)
            with st.expander("Show Raw Extracted Info", icon="🖋️"):
                st.code(ocr_text)
