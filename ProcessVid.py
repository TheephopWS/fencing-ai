import os
import time
from PoseEstimation.Training import main 

def ProcessVid(path, upload_directory):
    time.sleep(5)

    processed_paths = main()

    video_filename = os.path.basename(path)
    processed_file_name = f'processed_{video_filename}'
    temp = os.path.join(upload_directory, processed_file_name)

    processed_path = os.path.join(os.path.dirname(__file__), temp)

    while not os.path.exists(processed_path):
        time.sleep(1)  # Wait for 1 second before checking again

    # Open the processed file when it exists
    with open(processed_path, "rb") as f_in:
        time.sleep(2)
        return os.path.normpath(processed_path)