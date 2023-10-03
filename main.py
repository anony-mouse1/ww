## This sample code is from https://www.twilio.com/docs/stun-turn/api
# Download the helper library from https://www.twilio.com/docs/python/install

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Then, pass the ICE server information to webrtc_streamer().
# Download the helper library from https://www.twilio.com/docs/python/install
import os
# from twilio.rest import Client

from streamlit_webrtc import webrtc_streamer

#configure STUN server 
webrtc_streamer(key="sample", rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]
    })

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure

# account_sid = os.environ['TWILIO_ACCOUNT_SID']
# auth_token = os.environ['TWILIO_AUTH_TOKEN']
# client = Client(account_sid, auth_token)

# token = client.tokens.create()

# Then, pass the ICE server information to webrtc_streamer().


# webrtc_streamer(
#   # ...
#   rtc_configuration={
#       "iceServers": token.ice_servers
#   }
#   # ...
# )


# import logging
# import queue
# from sample_utils.turn import get_ice_servers




# # Then, pass the ICE server information to webrtc_streamer().
# webrtc_streamer(
#   key="webcam",
#   rtc_configuration={
#       "iceServers": get_ice_servers()
#   }
#   # ...
# )

# webrtc_streamer(key="snapshot", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer,client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
