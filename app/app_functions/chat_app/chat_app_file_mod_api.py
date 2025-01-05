import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)



# Function to write updated content back to the .txt file
def write_txt_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


# to API
# Function to delete the .txt and .py files
def delete_files(base_name, directory):
    txt_file_path = os.path.join(directory, f"{base_name}.txt")
    py_file_path = os.path.join(directory, f"{base_name}.py")
    
    if os.path.exists(txt_file_path):
        os.remove(txt_file_path)
    if os.path.exists(py_file_path):
        os.remove(py_file_path)

# to API
# Function to copy a file using os and open
def copy_file_os(src, dest):
    with open(src, 'rb') as fsrc:  # Open source file in binary mode
        with open(dest, 'wb') as fdest:  # Open destination file in binary mode
            fdest.write(fsrc.read())  # Read the source file and write it to the destination

# to API
# Function to add a new prompt with the base name by copying template files
def add_new_prompt(base_name, directory):
    # Paths to template files
    print("starting adding new..")
    template_txt_path =  os.path.join(directory, "chat_app/template.txt")
    template_py_path = os.path.join(directory, "chat_app/template.py")
    print(template_txt_path)
    # Destination paths for the new files
    new_txt_file_path = os.path.join(directory, f"{base_name}.txt")
    new_py_file_path = os.path.join(directory, f"{base_name}.py")
    print(new_txt_file_path)
    # Copy template files to the new location with the new base name
    copy_file_os(template_txt_path, new_txt_file_path)
    copy_file_os(template_py_path, new_py_file_path)
    print("Added new file")

# to API
# Function to rename files (both .txt and .py)
def rename_files(old_base_name, new_base_name, directory):
    old_txt_file_path = os.path.join(directory, f"{old_base_name}.txt")
    old_py_file_path = os.path.join(directory, f"{old_base_name}.py")
    
    new_txt_file_path = os.path.join(directory, f"{new_base_name}.txt")
    new_py_file_path = os.path.join(directory, f"{new_base_name}.py")
    
    # Rename both files
    os.rename(old_txt_file_path, new_txt_file_path)
    os.rename(old_py_file_path, new_py_file_path)



# Route to write updated content back to a .txt file
@app.route('/write_txt', methods=['POST'])
def api_write_txt():
    data = request.get_json()
    file_name = data.get('file_name')
    content = data.get('content')
    directory = data.get('directory')
    file_path = os.path.join(directory, f"{file_name}.txt")
    print(file_path)
    if os.path.exists(file_path):
        write_txt_file(file_path, content)
        return jsonify({'message': 'File updated successfully'})
    else:
        return jsonify({'error': 'File not found'}), 404

# Route to delete .txt and .py files
@app.route('/delete_files', methods=['DELETE'])
def api_delete_files():
    data = request.get_json()
    base_name = data.get('base_name')
    directory = data.get('directory')
    delete_files(base_name, directory)
    return jsonify({'message': f"Files {base_name}.txt and {base_name}.py have been deleted"})

# Route to add a new prompt by copying template files
@app.route('/add_new_prompt', methods=['POST'])
def api_add_new_prompt():
    data = request.get_json()
    base_name = data.get('base_name')
    directory = data.get('directory')
    add_new_prompt(base_name, directory)
    return jsonify({'message': f"New prompt {base_name} created successfully with .txt and .py files"})

# Route to rename .txt and .py files
@app.route('/rename_files', methods=['POST'])
def api_rename_files():
    data = request.get_json()
    old_base_name = data.get('old_base_name')
    new_base_name = data.get('new_base_name')
    directory = data.get('directory')
    rename_files(old_base_name, new_base_name, directory)
    return jsonify({'message': f"Files renamed from {old_base_name} to {new_base_name}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
