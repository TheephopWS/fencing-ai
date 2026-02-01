import streamlit as st
import os
import time
from ProcessVid import ProcessVid
from utils.ClearFiles import ClearFiles

UPLOAD_DIRECTORY = 'uploads'
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

if __name__ == "__main__":
    st.title("Fencing AI App")

    getFile = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])

    if getFile is not None:
        path = os.path.join(UPLOAD_DIRECTORY, getFile.name)
        
        with open(path, "wb") as f:
            f.write(getFile.getbuffer())
        
        with st.spinner("Processing video..."):
            processed_path = ProcessVid(path, UPLOAD_DIRECTORY)

            time.sleep(2)

        st.success("Processing complete!")
        

        # os.system('ffmpeg -i {} -vcodec libx264 {}'.format(processed_path, processed_path.replace('tmp', )))
        st.video(processed_path)

    if st.button("Delete Old Files"):
        ClearFiles(UPLOAD_DIRECTORY)
        st.success("Old files deleted successfully!")



        