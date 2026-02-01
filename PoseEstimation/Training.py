import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Constants and model paths
start_time_seconds = 180
elapsed_frames = 0

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pose_landmarker_full.task')
VIDEO_SOURCE = 0
NUM_POSES = 2
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
VIDS_DIRECTORY = os.path.join(os.path.dirname(__file__), "../uploads")

VIDS_OUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "../uploads")
WIDTH = 640
HEIGHT = 480
ATT_BUFFER = 2
 
# Global variables
to_window = None
last_timestamp_ms = 0
frame_number = 0 
last_frame_number = -1
landmark_dict = {}
distance_info_previous = {}

# For Player attacking
attacker = 2 #0:left fencer, 1:right fencer, 2:no one
# attacker_score = {0:0, 1:0}

def isAttacking(image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2

    # Calculate text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Create a rectangle background
    background_color = (255, 0, 0)  # Red color for the background
    rectangle_start = position
    rectangle_end = (position[0] + text_size[0] + 10, position[1] - text_size[1] - 10)
    cv2.rectangle(image, rectangle_start, rectangle_end, background_color, cv2.FILLED)

    # Add text on top of the rectangle
    cv2.putText(image, text, position, font, font_scale, font_color, thickness)

def get_distance_info(detection_result):
    distance_info = {}
    pose_landmarks_list = detection_result.pose_landmarks
    for numPerson in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[numPerson]
        
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility = landmark.visibility) for landmark in pose_landmarks
        ])

        if numPerson == 0: # left fencer
            lead_foot = 27
            lead_hip = 23
            follow_hip = 28
            follow_foot = 24
            lead_distance = pose_landmarks_proto.landmark[lead_foot].x - pose_landmarks_proto.landmark[lead_hip].x
            follow_distance = pose_landmarks_proto.landmark[follow_hip].x - pose_landmarks_proto.landmark[follow_foot].x
        else: # right fencer
            lead_foot = 28
            lead_hip = 24
            follow_foot = 27
            follow_hip = 23
            lead_distance = pose_landmarks_proto.landmark[lead_hip].x - pose_landmarks_proto.landmark[lead_foot].x
            follow_distance = pose_landmarks_proto.landmark[follow_foot].x - pose_landmarks_proto.landmark[follow_hip].x
       
        feet_distance = abs(pose_landmarks_proto.landmark[lead_foot].x - pose_landmarks_proto.landmark[follow_foot].x)
        distance_info = distance_info | {numPerson:(feet_distance, lead_distance, follow_distance)}
    
    return distance_info

def get_attacking(distance_info, distance_info_previous={}):
    attacking = {0:0,1:0}
    if distance_info_previous=={}:
        return attacking
    for numPerson in [0,1]:
        try:
            pre = distance_info_previous[numPerson]
            now = distance_info[numPerson]
            if now[0]-pre[0]>0: 
                if now[1]-pre[1] > now[2]-pre[2]: #attack
                    attacking[numPerson] = 1
            else:
                if now[1]-pre[1] < now[2]-pre[2]: #attack
                    attacking[numPerson] = 1
        except:
            print("distance_info error")
            pass
    return attacking

def get_attacker(attacking):
    global attacker
    if attacking=={0:0,1:0}:
        attacker=2
    elif attacking=={0:1,1:0}:
        if attacker==1:
            attacker=2
        else:
            attacker=0
    elif attacking=={0:0,1:1}:
        if attacker==0:
            attacker=2
        else:
            attacker=1
    return attacker



def draw_landmarks_on_image(rgb_image, detection_result):
    global attacker, elapsed_frames

    frame_rate = 30

    pose_landmarks_list = detection_result.pose_landmarks
    video_frame = np.copy(rgb_image)

    # Draw pose landmarks
    for numPerson in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[numPerson]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility) for landmark in pose_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            video_frame,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    lower_panel_height = 100
    lower_panel = np.ones((lower_panel_height, video_frame.shape[1], 3), dtype=np.uint8) * 255  # white background

    # === Colors
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    black = (0, 0, 0)

    # === Calculate dynamic time
    elapsed_frames += 1
    time_left = max(start_time_seconds - elapsed_frames // frame_rate, 0)
    minutes = time_left // 60
    seconds = time_left % 60
    time_str = f"{minutes:02d}:{seconds:02d}"

    cv2.rectangle(lower_panel, (0, 40), (video_frame.shape[1], 100), (0, 0, 0), -1)

    cv2.putText(lower_panel, "0", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, red, 3, cv2.LINE_AA)
    cv2.putText(lower_panel, ":", (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, white, 3, cv2.LINE_AA)
    cv2.putText(lower_panel, "0", (370, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, green, 3, cv2.LINE_AA)

    cv2.putText(lower_panel, time_str, (260, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, black, 4)


    if attacker == 0:
        cv2.putText(lower_panel, "Attack", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, yellow, 3, cv2.LINE_AA)
    elif attacker == 2:
        cv2.putText(lower_panel, "Attack", (480, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, yellow, 3, cv2.LINE_AA)

    full_frame = np.vstack((video_frame, lower_panel))

    return full_frame


def print_result(detection_result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    global distance_info_previous
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    to_window = cv2.cvtColor(
        draw_landmarks_on_image(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR
    )
    print(distance_info_previous)
    attacking = get_attacking(get_distance_info(detection_result), distance_info_previous)
    print(attacking)
    print(f"attacker: {get_attacker(attacking)}")
    print("----")
    
    distance_info_previous = get_distance_info(detection_result)

def main() -> None:
    global frame_number
    frame_number = 0
    base_options = mp.tasks.BaseOptions(model_asset_buffer=open(MODEL_PATH, "rb").read())
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        output_segmentation_masks=False,
        result_callback=print_result
    )
    
    distance_info_previous = {}
    # cap = cv2.VideoCapture(1)  # Use camera 1 for real-time capture
    cap = cv2.VideoCapture("http://172.27.132.75:8080/video") 

    
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Image capture failed.")
                break

            resized_image = cv2.resize(image, (WIDTH, HEIGHT))
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
            
            frame_number += 1
            print(frame_number)
            if to_window is not None:
                cv2.imshow("MediaPipe Pose Landmark", to_window)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()