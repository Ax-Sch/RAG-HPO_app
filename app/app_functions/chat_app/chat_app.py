# chat_app.py
import os
import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from app_functions.llm_connection.llm import get_llm, get_avail_models

def get_system_file_name(script_filename):
    base_name, ext = os.path.splitext(script_filename)
    new_filename = base_name + '.txt'
    print(new_filename)
    return(new_filename)

# Function to load system message from a file
def load_system_message(file_name):
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        print("System message file not found!")
        return "You are a helpful assistant."


# Core function to run the chat app
def run_chat_app(system_message_file, model_name="llama3.1"):
    st.set_page_config(layout="wide")
    model_name = st.selectbox(
        "Choose a Model:",
        options=get_avail_models(),
        index=0  # Default model
    )

    # Load system message
    system_message_content = load_system_message(system_message_file)

    # Editable text input for system message
    system_message = st.text_area("System Message:", value=system_message_content, height=150)
    
    if not "chat_mode" in st.session_state:
        st.session_state.chat_mode=True

    def user_input_element(chat_mode):
        if chat_mode:
            return st.chat_input("Your Input:")
        else:
            return st.text_area(label="Your Input:", key="user_input", height=250)

    def button(chat_mode):
        if chat_mode:
            return True
        else:
            return st.button("Send")
            
    # Display chat history
    def show_chat():
        if "chat_history" in st.session_state:
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.write(f":blue[**You:**] {message.content}")
                elif isinstance(message, AIMessage):
                    st.write(f":red[**AI:**] {message.content}")

    # User input
    user_input = user_input_element(st.session_state.chat_mode)
    if button(st.session_state.chat_mode) and user_input:
        # init llm:
        llm=get_llm(model_name)
        # Add user's message to the chat history
        if "chat_history" in st.session_state:
            st.session_state.chat_history.append(SystemMessage(content=system_message) )
        else:
            st.session_state.chat_history = [SystemMessage(content=system_message)]
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        # Get response from the model
        print(st.session_state.chat_history)
        response = llm(st.session_state.chat_history)
        # Add AI's response to the chat history
        st.session_state.chat_history.append(AIMessage(content=response.content))
        print(st.session_state.chat_history)
    if st.button("Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.button("None-chat mode"):
        st.session_state.chat_mode=False
        st.rerun()

    show_chat()
    
