import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from src.auth import load_authenticator
from src.bootstrap import load_resources

# This should be first before any other Streamlit output
st.set_page_config(page_title="Receipt Analyzer", layout="wide")

# Login widget
authenticator = load_authenticator()
# If the user is not authenticated, stop the app
try:
    auth = authenticator.login("main")
except Exception as e:
    st.error(e)

# All the authentication info is stored in the session_state
if st.session_state["authentication_status"]:
    # User is connected
    authenticator.logout("Logout", "main")
elif st.session_state["authentication_status"] == False:
    st.error("Username/password is incorrect")
    # Stop the rendering if the user isn't connected
    st.stop()
elif st.session_state["authentication_status"] == None:
    st.warning("Please enter your username and password")
    # Stop the rendering if the user isn't connected
    st.stop()


# Trigger resource loading
# Show loading spinner while initializing models
with st.spinner("Initializing models..."):
    _ = load_resources()

st.title("MRA - Mavi's Receipt Analyzer 🤖")
st.info(
    "Hi Boo, welcome to your own Receipt Analyzer! "
    "\n\n"
    "This app analyzes your receipts to help automate your expenses report, "
    "to ease the process of mapping your receipts to your financial records, "
    "to reduce your level of cursing against Curiox (and the world), and "
    "hopefully avoids getting your Credit Card revoked ever again (almost)."
    "\n\n"
    "<_ Use the sidebar to upload..."
)
