import cv2
import os 
from twilio.rest import Client


from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from streamlit_webrtc import webrtc_streamer

account_sid = os.environ['AC5f866da1acb4b18b1adc169c5d540c90']
auth_token = os.environ['e9d37da9e2d87f28ef34cd426120a5de']
client = Client(account_sid, auth_token)

token = client.tokens.create()

webrtc_streamer(key="sample", rtc_configuration={"iceServers":token.ice_servers})

# webrtc_streamer(key="snapshot", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer,client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
