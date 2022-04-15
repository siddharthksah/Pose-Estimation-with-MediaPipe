import cv2
import mediapipe as mp
import numpy as np

import cv2
import streamlit as st

favicon = './files/icon.jpeg'

# st.set_page_config(page_title='Biceps Curl', page_icon = favicon, initial_sidebar_state = 'auto')

# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)


st.header("How many biceps curls can you actually do!")
st.write("\n")


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

if st.button("Let's go!💪💪", key=1):
    # run = st.checkbox('Run')
    FRAME_WINDOW = st.image([], use_column_width=True)
    
    cap = cv2.VideoCapture(0)

    ## Setup mediapipe instance
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            
            # _, frame = cap.read()
            (h, w ,_) = frame.shape

            scale_percent = 60 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100) 
            height = int(frame.shape[0] * scale_percent / 100) 
            dim = (width, height) 

            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            # frame = cv2.resize(frame, (224, 224))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Recolor image to RGB
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            out.write(image)
            # cv2.imshow('Mediapipe Feed', image)
            FRAME_WINDOW.image(image)
        
        except:
            pass
        
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
import pandas as pd
dataframe = np.array([["Name", "Score"],
                ["DiscoNinja","12"],
                ["PUBGod","8"],
                ["GreekYogurt","32"],
                ["PuppetMaster","1"],
                ["Lord","33"]])

dataframe = pd.DataFrame(dataframe[1:,1:], index=dataframe[1:,0], columns=dataframe[0,1:])

st.dataframe(dataframe)

st.success("Highest Score by Lord!!")