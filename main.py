## This sample code is from https://www.twilio.com/docs/stun-turn/api
# Download the helper library from https://www.twilio.com/docs/python/install

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Then, pass the ICE server information to webrtc_streamer().
webrtc_streamer(
  key="webcam",
  rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
  # ...
)



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
