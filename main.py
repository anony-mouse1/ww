import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from streamlit_webrtc import webrtc_streamer

webrtc_streamer(key="sample")

# webrtc_streamer(key="snapshot", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer,client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
