import cv2
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import mediapipe as mp
import paho.mqtt.client as mqtt

####################

#############################
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
############################
############################
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
###########################
def on_message(client, userdata, msg):
    global request_counter
    # Assuming the image is sent as a byte array
    image = msg.payload
    # Save the image to a file
    with open('C:/Users/FreeComp/Desktop/stem_projects/sign_lang(malak khaled)/ActionDetectionforSignLanguage/joo.jpg', 'wb') as image_file:
        image_file.write(image)
        print("Image received and saved to 'received_image.jpg'")
    request_counter +=1

################

#########################

###########################


            # Viz probabilities
#             image = prob_viz(res, actions, image, colors)
            
#         cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
#         cv2.putText(image, ' '.join(sentence), (3,30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
#         cv2.imshow('OpenCV Feed', image)

        # Break gracefully


if __name__ == '__main__':
    server_ip="192.168.78.179"
    while 1:  
        request_counter = 0
        max_requests = 1

        client = mqtt.Client()        
        client2 = mqtt.Client()        

        # Assign the callback function
        client.on_message = on_message
        

        # Connect to the broker
        client.connect(server_ip)  # Change this to your MQTT broker IP/hostname
        client.subscribe("photo")
        # Subscribe to the topic

        # Change this to your topic

        client.loop_start()

        try:
                # Keep the script running
            while request_counter < max_requests:
                    pass

        except KeyboardInterrupt:
                # Handle KeyboardInterrupt
                print("KeyboardInterrupt detected.")

        finally:
                # Cleanup actions
                client.loop_stop()  
                client.disconnect()
                print("Disconnected from MQTT broker")
        actions = np.array(['hello', 'thanks', 'iloveyou'])
        mp_holistic = mp.solutions.holistic # Holistic model

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1,1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))

        ############################
        model.load_weights('action1.h5')
        sequence = []
        sentence = []
        threshold = 0.8
        i=0
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                # Read feed
                
                img = cv2.imread('C:/Users/FreeComp/Desktop/stem_projects/sign_lang(malak khaled)/ActionDetectionforSignLanguage/joo.jpg')
                # Make detections
                image, results = mediapipe_detection(img, holistic)

                
                # Draw landmarks
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-1:]
                
                if len(sequence) == 1:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    client2.connect(server_ip)
                    client2.publish("sign",str(actions[np.argmax(res)]))

                    
                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                        break
                
                