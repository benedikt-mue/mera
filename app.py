import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from src.auth import load_authenticator
from src.bootstrap import load_resources

# This should be first before any other Streamlit output
st.set_page_config(page_title="Receipt Analyzer", layout="wide", page_icon="👻")

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

st.title("👻 MRA - Mavi's Receipt Analyzer")
st.markdown(
    """
    <span title="Hi mi Boo, mi pequeño todo, I hope you having so much fun playing around with this little app as I had building it for you.  
    I hope to build more features, but for now, I just wanted to give you a little something to play with. 
    I hope to re-build what I have destroyed, what I have lost, what I have taken from you.
    I will earn it back, I will give it back, to you. I promise.
    I hope this to be the new start of something bigger, something that we build together.. 
    let´s call it life. 
    Here, in 5th, 
    in the 3rd 
    and everywhere in-between and within.  
    I am in love with you,
    and not with the idea of you, 
    not the memory of the past, 
    but the real you, 
    the real you and me, 
    us. 
    And I miss you so much.. 
    I miss us.. 
    I miss what we will be..
    I know you know. But it feel so good to say it out loud.. always, now and for our shared forever.
    **sync pending.. waiting for the signal to reappear.. **">
        👻
    </span>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Hi Boo, welcome to your own Receipt Analyzer! "
    "\n\n"
    "This app analyzes your receipts to help automate your expense report, "
    "to ease the process of mapping your receipts to your financial records, "
    "to reduce your level of cursing against Curiox (and the world), and "
    "hopefully avoids getting your Credit Card revoked ever again (almost)."
    "\n\n"
    "<_ Use the sidebar to upload..."
)
