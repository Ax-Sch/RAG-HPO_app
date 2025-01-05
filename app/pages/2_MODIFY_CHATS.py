from app_functions.chat_app.chat_app_manage import *
import streamlit as st


# Define the base URL for the API (host is "api" on the network)
BASE_URL = "http://chat_app_file_mod_api:5000"

# directory with chat pages / prompts
directory = "app/pages/"  # Replace with the actual directory

# Get base filenames that have both .txt and .py files
base_filenames = get_base_filenames(directory)


# Streamlit UI
st.title("File Content Editor")

# Dropdown to select a base filename
selected_base_name = st.selectbox("Select a file:", base_filenames)

# Load the content of the corresponding .txt file
txt_file_path = os.path.join(directory, f"{selected_base_name}.txt")
file_content = read_txt_file(txt_file_path)

# Display the content in a text area for editing
updated_content = st.text_area("Edit the content:", value=file_content, height=300)

enable_edit = st.checkbox("Activate the buttons below:")

# Update button to save the changes
if st.button("Update Text") and enable_edit:
    write_txt_file(selected_base_name, updated_content, directory)
    st.success(f"File '{selected_base_name}.txt' has been updated successfully!")

# Update Name button
new_name = st.text_input(label="Enter a new name to rename the selected file:", value=selected_base_name )
if st.button("Update Name") and new_name and enable_edit:
    if new_name not in base_filenames:  # Ensure the new name doesn't already exist
        rename_files(selected_base_name, new_name, directory)
        st.success(f"Files have been renamed from '{selected_base_name}' to '{new_name}.txt' and '{new_name}.py'.")
        base_filenames = get_base_filenames(directory)  # Refresh the base filenames list
    else:
        st.error("The new name already exists. Please choose a different name.")


# Remove button to initiate the deletion process
if st.button("Remove") and enable_edit:
    # Display confirmation warning
    delete_files(selected_base_name, directory)
    st.success(f"Files {selected_base_name}.txt and '{selected_base_name}.py have been removed.")
 

new_base_name = st.text_input("Enter a new base name for a new prompt:")
# Add New button
if st.button("Add New") and enable_edit:
    if new_base_name not in base_filenames:  # Ensure the name doesn't already exist
        print("adding new chat")
        add_new_prompt(new_base_name, directory)
        st.success(f"New files '{new_base_name}.txt' and '{new_base_name}.py' have been created.")
        base_filenames = get_base_filenames(directory)  # Refresh the base filenames list
    else:
        st.error("The base name already exists. Please choose a different name.")

