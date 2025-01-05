import json
import csv
import re
import sys
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from fastembed import TextEmbedding

# Increase the CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

# Paths to the files
JSON_FILE_PATH = 'hpo_data_with_lineage.json'
CSV_FILE_PATH = 'HPO_addons.csv'
OUTPUT_FILE = 'G2GHPO_metadata_test.npy'
MODEL_NAME = "BAAI/bge-small-en-v1.5"
CSV_OUTPUT_FILE = 'HP_DB_test.csv'  # Output file for manual inspection

# Load the CSV data
csv_data = pd.read_csv(CSV_FILE_PATH)

# Regular expression pattern to remove parentheses and their contents
PATTERN = re.compile(r'\(.*?\)')

def clean_text(text):
    return re.sub(PATTERN, '', text).replace('_', ' ').lower()

def process_json_file(json_file_path, csv_data):
    """Processes the JSON file and integrates additional information from the CSV."""
    data = []
    csv_rows = []  # Rows for the CSV output
    with open(json_file_path, 'r') as file:
        hpo_data = json.load(file)
    for hp_id, details in hpo_data.items():
        formatted_hp_id = hp_id.replace('_', ':')
        unique_info = set()

        # Clean and add label
        label = details.get('label')
        if label:
            unique_info.add(clean_text(label))

        # Clean and add synonyms
        synonyms = details.get('synonyms', [])
        for synonym in synonyms:
            unique_info.add(clean_text(synonym))

        # Clean and add definition
        definition = details.get('definition', '')
        if definition:
            unique_info.add(clean_text(definition))

        # Add CSV info if available
        csv_addons = csv_data[csv_data['HP_ID'] == formatted_hp_id]['info'].tolist()
        for addon in csv_addons:
            unique_info.add(clean_text(addon))

        # Include lineage information
        lineages = details.get('lineage', [])
        for info in unique_info:
            data.append((formatted_hp_id, info, ', '.join(lineages)))
            csv_rows.append({'HP_ID': formatted_hp_id, 'info': info, 'lineage': ', '.join(lineages)})
    return data, csv_rows

def save_to_csv(csv_rows, output_file):
    """ Saves processed data to a CSV file for manual inspection."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['HP_ID', 'info', 'lineage'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Processed data has been written to {output_file} for inspection.")

def calculate_depth(lineage):
    """ Calculates depth of a term based on the lineage hierarchy. """
    return lineage.count("->") + 1

def extract_organ_system(lineage):
    """ Extracts the organ system from the lineage hierarchy. """
    parts = lineage.split("->")
    return parts[1].strip() if len(parts) > 1 else "Unknown"

def create_vector_database(data, output_file, model_name):
    """ Embeds structured data and saves the embeddings and metadata."""
    print("Initializing embeddings model...")
    embedding_model = TextEmbedding(model_name=model_name)
    if os.path.exists(output_file):
        print("Loading existing embedded documents...")
        embedded_documents = list(np.load(output_file, allow_pickle=True))
    else:
        print("Starting with new embedded documents list...")
        embedded_documents = []
    print(f"Data prepared with {len(data)} terms to embed.")
    batch_size = 100
    total_batches = (len(data) + batch_size - 1) // batch_size
    print("Starting the embedding process...")
    for i in tqdm(range(0, len(data), batch_size), total=total_batches, desc="Embedding Texts"):
        batch_data = data[i:i + batch_size]
        for hp_id, cleaned_info, lineage in batch_data:
            try:
                depth = calculate_depth(lineage)
                organ_system = extract_organ_system(lineage)
                embedding = np.array(list(embedding_model.embed([cleaned_info]))[0])
                metadata = {
                    'embedding': embedding,
                    'metadata': {'info': cleaned_info, 'hp_id': hp_id},
                    'lineage': lineage,
                    'organ_system': organ_system,
                    'depth_from_root': depth
                }
                embedded_documents.append(metadata)
            except Exception as e:
                print(f"Failed to embed text due to {e}")
    np.save(output_file, embedded_documents, allow_pickle=True)
    print(f"All embeddings and metadata are saved in: {output_file}")

def main():   
    # Process JSON and integrate CSV data
    data, csv_rows = process_json_file(JSON_FILE_PATH, csv_data)

    # Save the processed data to a CSV for manual inspection
    save_to_csv(csv_rows, CSV_OUTPUT_FILE)
    print(f"The database contains {len(data)} entries ready for embedding.")
    # Embed data and save the vector database
    create_vector_database(data, OUTPUT_FILE, MODEL_NAME)

if __name__ == "__main__":
    main()