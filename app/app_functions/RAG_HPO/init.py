import requests
import os


def download_file(url, output_path):
    """
    Downloads a file from the given URL and saves it to the specified output path.

    Parameters:
        url (str): The URL of the file to download.
        output_path (str): The path where the file should be saved.

    Returns:
        bool: True if the file was downloaded successfully, False otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open(output_path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded and saved as {output_path}")
        return True
    except requests.RequestException as e:
        print(f"Failed to download file: {e}")
        return False


script_path=os.path.realpath(__file__)
script_dir=os.path.dirname(script_path)

os.chdir(script_dir)

# get hp.json
HP_JSON_DL_LINK="https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-12-12/hp.json"

print("Download HP.json from " + HP_JSON_DL_LINK)

download_file(HP_JSON_DL_LINK, "hp.json")

print("Running first part of vectorization")
os.system("python3 HPO_Vectorization_1.py")

print("Running second part of vectorization")
os.system("python3 HPO_Vectorization_2.py")
