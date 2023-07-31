import pandas as pd
import moviepy.editor
import numpy as np
import cv2
import mediapipe as mp
# import test_model as tm
import time
import multiprocessing as mpg
import tensorflow as tf
from moviepy.editor import VideoFileClip
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
l = [0]
# For webcam input:
pq_file = ".\Try.parquet"
xyz = pd.read_parquet(pq_file)
xyz_skel = (xyz[["type", "landmark_index"]].drop_duplicates().reset_index(drop = True).copy())
def create_frame_landmark_df(results, frame):
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    face = (face.reset_index().rename(columns = {"index":"landmark_index"}).assign(type = "face"))
    pose = (pose.reset_index().rename(columns = {"index":"landmark_index"}).assign(type = "pose"))
    left_hand = (left_hand.reset_index().rename(columns = {"index":"landmark_index"}).assign(type = "left_hand"))
    right_hand = (right_hand.reset_index().rename(columns = {"index":"landmark_index"}).assign(type = "right_hand"))
    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop = True)
    landmarks = xyz_skel.merge(landmarks, on = ["type", "landmark_index"], how = "left")
    landmarks = landmarks.assign(frame = frame)
    return landmarks
# video = cv2.VideoCapture('D:/Downloads/NUS/Project/Capture3.mp4')
# def video2():
    # pool = mpg.Pool(mpg.cpu_count())
    # # Step 2: `pool.apply` the `howmany_within_range()`
    # list = []
    # results = [pool.map(main_loop, list)]
    # # print(results)
    # # Step 3: Don't forget to close
    # pool.close()
def video_capture():
# Check if camera opened successfully
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_name = 1
    start = time.time()
    end = start + 12
    while(time.time() < end):
        time.sleep(0.05)
        start_time = time.time()
        end_time = start_time + 4
        if (cap.isOpened() == False): 
            print("Camera is unable to open.")

        # Set resolutions of frame.
        # convert from float to integer.

        # Create VideoWriter object.
        # and store the output in 'captured_video.avi' file.
        video_cod = cv2.VideoWriter_fourcc(*'XVID')
        x = "./temp/" + "captured_vid" + str(video_name) + ".avi" 
        video_output= cv2.VideoWriter(x, video_cod, 30, (frame_width, frame_height))
        video_name += 1
        while(True):
            ret, frame = cap.read()
            cv2.putText(frame, str(int(time.time() - start_time)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),2,cv2.LINE_AA)
            if ret == True: 
                
                # Write the frame into the file 'captured_video.avi'
                video_output.write(frame)
                # Display the frame, saved in the file   
                cv2.imshow('Recording Video...',frame)
                
                # Press x on keyboard to stop recording
                if (cv2.waitKey(5) & 0xFF == 27) or time.time() > end_time:
                    video_output.release()
                    break
            # Break the loop
            else:
                break
        
        # video = cv2.VideoCapture("D:/Downloads/NUS/Project/captured_video.avi")
        # print(x)
        l.append(l[len(l) - 1] + 1)
        x = [l[len(l) - 1]]
        process = mpg.Process(target=main_loop, args=x)
        process.start()
        if((cv2.waitKey(5) & 0xFF == 27)):
            cap.release()
            video_output.release()
            break
    # process.terminate()
    cv2.destroyAllWindows()
def main_loop(x: list):
    # predicted= []
    vid = "./temp/captured_vid" + str(x) + ".avi"
    video = cv2.VideoCapture(vid)
    clip = VideoFileClip(vid)
    duration = clip.duration
    if(duration < 3):
        return 0
    time_for_each_sign = 4
    # video = cv2.VideoCapture('D:/Downloads/NUS/Project/Capture3.mp4')
    # fps = video.get(cv2.CAP_PROP_FPS)
    # print(fps)
    duration = 1
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        curr_frame = 0
        while duration > 0:
            all_landmarks = []
            frame = 0
            frame_count = 24
            x = (time_for_each_sign*30)//frame_count
            while frame_count > 0:
                video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                curr_frame += x
                frame_count -= 1
                ret, image = video.read()
                frame += 1
                if ret == True:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # name = "image" + str(frame) + '.png'
                    # cv2.imwrite(name, image)
                    results = holistic.process(image)
                    #Create landmark dataframe
                    landmarks = create_frame_landmark_df(results, frame)
                    all_landmarks.append(landmarks)
            if(len(all_landmarks) > 0):
                all_landmarks = pd.concat(all_landmarks).reset_index(drop = True).to_parquet('./temp/output.parquet')
                x = Interpreter()
                with open("./temp/text.txt","a") as file:
                    file.write(x)
                    file.write('\n')
            duration -= 1
    # os.remove(vid)
    # print(all_landmarks)

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) /543)
    data = data.values.reshape(n_frames, 543, len(data_columns))
    return data.astype(np.float32)
def Interpreter():
    interpreter = tf.lite.Interpreter("./model2.tflite")
    prediction_fn = interpreter.get_signature_runner("serving_default")
    found_signatures = list(interpreter.get_signature_list().keys())
    train = pd.read_csv("./train.csv")
    train['sign_ord'] = train['sign'].astype('category').cat.codes

    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    # pq_file = 'D:/Downloads/NUS/Project/Asl-signs/Train_landmark_files/26734/1000035562.parquet'  
    pq_file = "./temp/output.parquet"
    xyz_np = load_relevant_data_subset(pq_file)
    prediction = prediction_fn(inputs=xyz_np)
    sign = prediction['outputs']
    sign = prediction['outputs'].argmax()
    os.system('cls')
    return ORD2SIGN[sign]

if __name__ == "__main__":
    with open("./temp/text.txt", "w") as file:
        file.write('')
    video_capture()