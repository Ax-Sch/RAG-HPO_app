import os
import requests
import json

BASE_URL = "http://chat_app_file_mod_api:5000"


# Function to get base filenames with both .txt and .py files in the directory
def get_base_filenames(directory):
    filenames = os.listdir(directory)
    base_names = set()  # Using a set to avoid duplicates
    
    # Iterate over the files to check if both .txt and .py files exist for the same base name
    for filename in filenames:
        if filename.endswith(".txt"):
            base_name = filename[:-4]  # Remove .txt extension
            py_file = base_name + ".py"
            if py_file in filenames:  # Check if the corresponding .py file exists
                base_names.add(base_name)
                
    return list(base_names)


# Function to read the content of a .txt file
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Function to write content to a .txt file
def write_txt_file(file_name, content, directory):
    url = f"{BASE_URL}/write_txt"
    data = {
        'file_name': file_name,
        'content': content,
        'directory': directory
    }
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print("File updated successfully.")
    else:
        print(f"Error: {response.json()['error']}")


# Function to delete .txt and .py files
def delete_files(base_name, directory):
    url = f"{BASE_URL}/delete_files"
    data = {'base_name': base_name, 'directory': directory}
    response = requests.delete(url, json=data)
    
    if response.status_code == 200:
        print(f"Files {base_name}.txt and {base_name}.py have been deleted.")
    else:
        print(f"Error: {response.json()['error']}")


# Function to add a new prompt by copying template files
def add_new_prompt(base_name, directory):
    url = f"{BASE_URL}/add_new_prompt"
    data = {'base_name': base_name, 'directory': directory}
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print(f"New prompt {base_name} created successfully.")
    else:
        print(f"Error: {response.json()['error']}")


# Function to rename .txt and .py files
def rename_files(old_base_name, new_base_name, directory):
    url = f"{BASE_URL}/rename_files"
    data = {
        'old_base_name': old_base_name,
        'new_base_name': new_base_name,
        'directory': directory
    }
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print(f"Files renamed from {old_base_name} to {new_base_name}.")
    else:
        print(f"Error: {response.json()['error']}")
