import os
import streamlit as st

def ClearFiles(UPLOAD_DIRECTORY):
    for file in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, file)
        try:
            os.remove(path)
        except Exception as e:
            st.error(f"Error deleting file {file}: {e}")