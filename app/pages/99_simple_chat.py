import os
import streamlit as st
from app_functions.chat_app.chat_app import *

# takes the current filename and exchanges the ending to .txt:
script_filename = __file__
system_message_file = get_system_file_name(script_filename) 

# Run the chat app with the specific system message file
run_chat_app(system_message_file)