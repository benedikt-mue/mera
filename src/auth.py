import logging

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

user_credentials_path = "user-credentials.yaml"


def load_authenticator():
    with open(user_credentials_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    # try:
    #     # Pre-hashing all plain text passwords once
    #     stauth.Hasher.hash_passwords(config["credentials"])

    #     # Save the Hashed Credentials to our config file
    #     with open(user_credentials_path, "w") as file:
    #         yaml.dump(config, file, default_flow_style=False)
    # except Exception as e:
    #     pass

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    return authenticator
