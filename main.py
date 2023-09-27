import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

#demo video 
# DEMO_VIDEO = 'ri1.mp4'



#mediapipe inbuilt solutions 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
st.set_page_config(layout="wide")


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


class VideoTransformer(VideoTransformerBase):
    def recv(self, frame): # i think self, frame --> ret, image
        image = frame.to_ndarray(format="rgb24")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

   
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose1, \
            mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose2:
            results = pose1.process(image) #previously image_rgb
    #         results2 = pose2.process(image2)

            if results.pose_landmarks:
                landmarks1 = results.pose_landmarks.landmark
                shoulder1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1]),
                         int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])]
                elbow1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1]),
                        int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])]
                wrist1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1]),
                      int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])]

                angle1 = calculate_angle(shoulder1, elbow1, wrist1)
                cv2.putText(image, f'Angle: {round(angle1, 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.circle(image, tuple(shoulder1), 15, (255, 255, 255), -1)
                cv2.circle(image, tuple(elbow1), 15, (255, 255, 255), -1)
                cv2.circle(image, tuple(wrist1), 15, (255, 255, 255), -1)

                if 32 <= abs(angle1) <= 175:
                    cv2.putText(image, 'YES', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(image, 'INCORRECT FORM', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
        return (image)



def main():

    #title 
    st.title('Workout Wiz')

    #sidebar title
    st.sidebar.title('Choose your exercise')

    # st.sidebar.subheader('Parameters')
    #creating a button for webcam
    use_webcam = st.sidebar.button('Use Webcam')
    #creating a slider for detection confidence 
    # detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    
    #model selection 
    model_selection = st.sidebar.selectbox('Model Selection',options=['Bicep Curls','Squats','Plank'])

    st.markdown(' ## Output')
    stframe = st.empty()

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)



    # st.sidebar.text('Input Video')
    # st.sidebar.video(tfflie.name)
    

    # original = Image.open(image)



    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# HEEE HEE
    # mp_drawing = mp.solutions.drawing_utils
    # mp_pose = mp.solutions.pose

    # col1, col2 = st.columns(2)

    # vid = cv2.VideoCapture(0)
    # instructor = cv2.VideoCapture('squat.mp4')

    # if instructor.isOpened():
    #     ret2, image2 = instructor.read()
        
    #     if ret2:
    #         with col2:
    #             st.video('squat.mp4')
    
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose1, \
    #     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose2:
        
    #     while vid.isOpened() and instructor.isOpened():
            
    #         ret, image = vid.read()
    #         resized_frame = cv2.resize(image, (600,800))
                   
    #         # ret2, image2 = instructor.read()

  
    #         if not ret:
    #             break
            
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    #         results = pose1.process(image) #previously image_rgb
    #         results2 = pose2.process(image2)

    #         if results.pose_landmarks:
    #             landmarks1 = results.pose_landmarks.landmark
    #             shoulder1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1]),
    #                      int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])]
    #             elbow1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1]),
    #                     int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])]
    #             wrist1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1]),
    #                   int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])]

    #             angle1 = calculate_angle(shoulder1, elbow1, wrist1)
    #             cv2.putText(image, f'Angle: {round(angle1, 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #             cv2.circle(image, tuple(shoulder1), 15, (255, 255, 255), -1)
    #             cv2.circle(image, tuple(elbow1), 15, (255, 255, 255), -1)
    #             cv2.circle(image, tuple(wrist1), 15, (255, 255, 255), -1)

    #             if 32 <= abs(angle1) <= 175:
    #                 cv2.putText(image, 'YES', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    #             else:
    #                 cv2.putText(image, 'INCORRECT FORM', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
    #         with col1:
                
    #             stframe.image(image,use_column_width=True)

            
        
        
        
    
        

    #     vid.release()
    #     # instructor.release()
    #     # out.release()
    #     cv2.destroyAllWindows()

# HEE HEE END.

if __name__ == '__main__':
    main()