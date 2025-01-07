from contextlib import contextmanager
from datetime import datetime
from fastembed import TextEmbedding
from fuzzywuzzy import fuzz
from langchain_core.prompts import ChatPromptTemplate
from requests.exceptions import HTTPError
from ratelimit import limits, sleep_and_retry
from tabulate import tabulate
from tqdm import tqdm
from app_functions.llm_connection.llm import initialize_embeddings_model, LLMClient, initalize_local_ollama_environment

import requests
import faiss
import json
import numpy as np
import os
import pandas as pd
import re
import string
import subprocess
import time
import sys
import pickle
import ast
import socket

rag_hpo_dir = os.path.dirname(__file__) + "/"
# set temp directory
os.environ['TMPDIR'] = '/python_tmp/'

with open(rag_hpo_dir + 'hpo_data_with_lineage.json', 'r') as file:
    hpo_data = json.load(file)

# Global Constants
FLAG_FILE = "process_completed.flag"
INITIALIZED = False  # Tracks if the function has been called
TOTAL_TOKENS_USED = 0  # Initialize token usage tracking globally
MAX_QUERIES_PER_MINUTE = 30
MAX_TOKENS_PER_DAY = 500000
MAX_QUERIES_PER_DAY = MAX_QUERIES_PER_MINUTE * 60 * 24

def load_prompts(file_path= rag_hpo_dir + "system_prompts.json"):
    with open(file_path, "r") as file:
        return json.load(file)
    
# Load prompts
prompts = load_prompts()

# Access prompts when needed
system_message_I = prompts["system_message_I"]
system_message_II = prompts["system_message_II"]



def generate_hpo_terms(df, system_message):
    """Generates HPO terms from a DataFrame of phrases and metadata."""
    responses = []
    for _, row in df.iterrows():
        user_input = row['phrase']
        unique_metadata_list = row['unique_metadata']
        original_sentence = row['original_sentence']
        
        # Prepare the human message with context
        context_items = []
        for item in unique_metadata_list:
            parsed_item = clean_and_parse(item)
            if parsed_item:
                for description, hp_id in parsed_item.items():
                    context_items.append(f"- {description} ({hp_id})")

        context_text = '\n'.join(context_items)
        human_message = (f"Query: {user_input}\n"
                         f"Original Sentence: {original_sentence}\n"
                         f"Context: The following related information is available to assist in determining the appropriate HPO terms:\n"
                         f"{context_text}")
        
        # Query the LLM using the unified function
        response_text = llm_client.query(human_message, system_message)
        
        # Extract HPO terms with regex
        hpo_terms = re.findall(r'HP:\d+', response_text)
        if hpo_terms:
            responses.append({"phrase": user_input, "response": ', '.join(hpo_terms)})
        else:
            responses.append({"phrase": user_input, "response": 'No HPO terms found'})
    
    return pd.DataFrame(responses)

# Context manager for subprocess handling
@contextmanager
def managed_subprocess(*args, **kwargs):
    """Manages a subprocess, ensuring it terminates properly upon completion or failure."""
    proc = subprocess.Popen(*args, **kwargs)
    try:
        yield proc
    finally:
        proc.terminate()  # Ensures proper termination of the subprocess
        proc.wait()

# Function for printing with timestamps
def timestamped_print(message):
    """Prints a message with the current timestamp for easy log tracking."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def load_embedded_documents(file_path):
    """Loads embedded documents (embeddings) from a given file path."""
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        exit(1)  # Exit if file not found
        print("Error: File not found.")

def prepare_embeddings_list(embedded_documents):
    """Converts the embedded documents into a NumPy array of embeddings."""
    embeddings_list = [np.array(doc['embedding']) for doc in embedded_documents if isinstance(doc['embedding'], np.ndarray) and doc['embedding'].size > 0]
    if not embeddings_list:
        exit(1)  # Exit if no valid embeddings found
    first_embedding_size = embeddings_list[0].shape[0]  # Ensure uniform embedding size
    return np.vstack([emb for emb in embeddings_list if emb.shape[0] == first_embedding_size])

def create_faiss_index(embeddings_array):
    """Creates a FAISS index for efficient similarity searching on embeddings."""
    if embeddings_array.dtype != np.float32:
        embeddings_array = embeddings_array.astype(np.float32)  # Ensure the correct data type for FAISS
    dimension = embeddings_array.shape[1]  # Determine embedding dimensionality
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
    index.add(embeddings_array)  # Add embeddings to FAISS index
    return index

def process_row(clinical_note, system_message_I, embeddings_model, index, embedded_documents):
    """Processes a single clinical note by extracting findings and matching metadata."""
    findings_text = llm_client.query(clinical_note, system_message_I)  # Extract findings from clinical note
    if not findings_text:
        return None  # If no findings, skip processing
    findings = extract_findings(findings_text)  # Extract findings from the note
    if not findings:
        return None  # Skip processing if no findings are extracted
    results_df = process_findings(findings, clinical_note, embeddings_model, index, embedded_documents)  # Process and match findings to metadata
    return results_df

def extract_findings(response_content):
    """Extracts findings (key information) from the response content generated by the LLM."""
    try:
        data = json.loads(response_content)  # Parse the JSON content
        findings = data.get("findings", [])
        return findings
    except json.JSONDecodeError:
        return []  # Return an empty list if the content cannot be parsed

def process_findings(findings, clinical_note, embeddings_model, index, embedded_documents):
    """Matches findings with their most relevant metadata entries from embeddings."""
    results = []
    # Split the clinical note into sentences for matching findings to specific contexts
    sentences = clinical_note.split('.')
    
    for finding in findings:
        # Embed the query phrase (finding) using the embeddings model
        query_vector = np.array(list(embeddings_model.embed([finding]))[0]).astype(np.float32).reshape(1, -1)
        distances, indices = index.search(query_vector, 800)  # Search for the most similar embeddings
        
        seen_metadata = set()
        unique_metadata = []

        for idx in indices[0]:
            # Retrieve metadata for the matched embedding
            #print(idx)
            #print(embedded_documents)
            metadata = embedded_documents[idx]['metadata']
            metadata_str = json.dumps(metadata)  # Convert the metadata dict to a string

            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)  # Track unique metadata
                unique_metadata.append(metadata_str)
                if len(unique_metadata) == 20:  # Limit to the first 20 unique metadata items
                    break

        # Find the best matching sentence from the clinical note for the finding
        finding_words = set(re.findall(r'\b\w+\b', finding.lower()))
        best_match_sentence = None
        max_matching_words = 0

        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            common_words = finding_words & sentence_words

            if len(common_words) > max_matching_words:
                max_matching_words = len(common_words)
                best_match_sentence = sentence.strip()  # Store the sentence with the most matching words

        # Store the results for this finding
        results.append({
            "phrase": finding,
            "unique_metadata": unique_metadata,  # Save unique metadata for this finding
            "original_sentence": best_match_sentence})  # Save the best matching sentence 

    # Convert the results to a DataFrame and save as a CSV file
    # The resulting CSV file will contain the extracted findings and their matched metadata 
    faiss_results_df = pd.DataFrame(results)
    #faiss_results_df.to_csv('faiss_search_results.csv', index=False)
    return faiss_results_df  # Return the DataFrame of results

# Helper functions for text cleaning and metadata processing
#Cleans and parses a JSON string by fixing formatting issues.
def clean_and_parse(json_str):
    try:
        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
        json_str = re.sub(r'\s+', ' ', json_str)  # Remove excess whitespace
        return json.loads(json_str)  # Return parsed JSON
    except json.JSONDecodeError:
        return None  # Return None if the string cannot be parsed

def process_unique_metadata(metadata):
    """Processes unique metadata by converting all keys to lowercase."""
    if isinstance(metadata, list):
        processed_list = []
        for item in metadata:
            try:
                item_dict = json.loads(item)  # Convert string back to dictionary
                processed_item = {k.lower(): v for k, v in item_dict.items()}  # Make keys lowercase
                processed_list.append(json.dumps(processed_item))  # Convert back to string
            except (json.JSONDecodeError, TypeError):
                continue
        return processed_list  # Return the list of processed metadata
    else:
        return []

def clean_text(text):
    """Cleans input text by converting to lowercase and removing punctuation."""
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).strip()  # Remove punctuation and trim whitespace
    return text

def extract_hpo_term(phrase, metadata_list):
    """ Extracts HPO terms by matching phrases against a list of metadata. The metadata is loaded from the unique_metadata field in the npy array."""
    cleaned_phrase = clean_text(phrase)  # Clean the input phrase
    fuzzy_matches = []

    # Step 1: Fuzzy Matching with Metadata
    for metadata in metadata_list:
        try:
            # Convert metadata string to dictionary if necessary
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            for term, hp_id in metadata_dict.items():
                cleaned_term = clean_text(term)  # Clean the metadata term
                if fuzz.ratio(cleaned_phrase, cleaned_term) > 80:  # Check for high similarity
                    fuzzy_matches.append({term: hp_id})
        except (json.JSONDecodeError, TypeError):
            continue  # Skip invalid metadata entries

    # If we have fuzzy matches, extend the list
    if fuzzy_matches:
        metadata_list.extend([json.dumps(match) for match in fuzzy_matches])

    # Step 2: Exact Substring Matching
    exact_matches = []
    for metadata in metadata_list:
        try:
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            for term, hp_id in metadata_dict.items():
                cleaned_term = clean_text(term)  # Clean the metadata term
                if cleaned_term in cleaned_phrase:  # Check if term is a substring of the phrase
                    exact_matches.append({term: hp_id})
        except (json.JSONDecodeError, TypeError):
            continue  # Skip invalid metadata entries

    # If we have exact matches, extend the list
    if exact_matches:
        metadata_list.extend([json.dumps(match) for match in exact_matches])

    # Step 3: Exact Matching within Metadata List
    for metadata in metadata_list:
        if not metadata.strip():
            continue
        try:
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            for term, hp_id in metadata_dict.items():
                cleaned_term = clean_text(term)
                if cleaned_phrase == cleaned_term:  # Check for an exact match
                    print(f"Exact match found: {hp_id}")
                    return hp_id  # Return the exact match
        except (json.JSONDecodeError, TypeError):
            continue  # Skip invalid metadata entries

    return None  # Return None if no match is found


# Process the results
def process_results(final_result_df):
    save_or_display = input("Do you want to save the results as a CSV file or display them? (save/display): ").strip().lower()
    
    if save_or_display == 'save':
        output_file = input("Enter the name of the output file (with .csv extension): ").strip()
        
        # Prepare the structured data for saving
        new_data = []
        for idx, row in final_result_df.iterrows():
            patient_id = row['patient_id']
            hpo_terms = row['HPO_Terms']  # Directly access the list of dictionaries
            
            # Process each term
            for term in hpo_terms:
                phrase = term.get('phrase', '').strip()
                hpo_id = term.get('HPO_Term', '').replace("HP:HP:", "HP:")
                
                if not hpo_id:
                    print(f"Blank HPO_Term for patient_id {patient_id} with phrase '{phrase}'")
                
                new_data.append({
                    'Patient_ID': patient_id,
                    'Phenotype name': phrase,
                    'HPO_ID': hpo_id
                })
        
        # Create a new DataFrame and save it
        new_df = pd.DataFrame(new_data)
        new_df.to_csv(output_file, index=False)
        json_csv_file = f"{output_file}_json.csv"
        final_result_df.to_csv(json_csv_file, index=False)
        timestamped_print(f"Data has been successfully saved to {output_file}")
    
    elif save_or_display == 'display':
        # Prepare the structured data for display
        flattened_data = []
        print(final_result_df)
        for idx, row in final_result_df.iterrows():
            patient_id = row['patient_id']
            hpo_terms = row['HPO_Terms']  # Directly access the list of dictionaries
            
            # Process each term
            for term in hpo_terms:
                flattened_data.append({
                    'Case': f"Case {patient_id}",
                    'Phenotype name': term.get('phrase', '').strip(),
                    'HPO ID': term.get('HPO_Term', '').replace("HP:HP:", "HP:")
                })
        
        if flattened_data:
            flattened_df = pd.DataFrame(flattened_data)
            print(tabulate(flattened_df, headers='keys', tablefmt='psql'))
        else:
            timestamped_print("No HPO terms found to display.")
    else:
        print("Invalid choice. Please choose either 'save' or 'display'.")
        
#if __name__ == "__main__":

def process_hpo_simple(final_result_df):
    # convert final results to table
    # add original HPO Term
    
    new_data = []
    for idx, row in final_result_df.iterrows():
        #patient_id = row['patient_id']
        hpo_terms = row['HPO_Terms']  # Directly access the list of dictionaries
        
        # Process each term
        for term in hpo_terms:
            phrase = term.get('phrase', '')
            hpo_ids = term.get('HPO_Term', '').replace("HP:HP:", "HP:").split(",")  # Ensure we split if multiple HPOs

            # Handle missing HPO_Term
            if not hpo_ids:
                print(f"Blank HPO_Term for patient_id {patient_id} with phrase '{phrase}'")
            
            # For each HPO ID (in case there are multiple)
            for hpo_id in hpo_ids:
                new_data.append({
                    #'Patient_ID': patient_id,
                    'Phenotype name': phrase,
                    'HPO_ID': hpo_id.strip()  # Strip any extra spaces from HPO IDs
                })
    result_df = pd.DataFrame(new_data)
    result_df = result_df.drop_duplicates()

    return(result_df)

def get_hpo_info(hpo_term):
    hpo_key = hpo_term.replace("HP:", "HP_")  # Adjust format to match the data keys
    hpo_info = hpo_data.get(hpo_key, None)
    if hpo_info:
        label = hpo_info.get('label', 'Label not found')
        synonyms = hpo_info.get('synonyms', [])
        return label, synonyms
    return None, None

def join_label_and_synonyms(phrases_data):
    for index, entry in phrases_data.iterrows():
        #print(entry)
        hpo_term = entry['HPO_ID']  # Use 'entry' to get the value for the current row
        label, synonyms = get_hpo_info(hpo_term)
        synonyms_str = ', '.join(synonyms) if isinstance(synonyms, list) else synonyms

        # Use .loc to modify the DataFrame efficiently
        phrases_data.loc[index, 'label'] = label
        phrases_data.loc[index, 'synonyms'] = synonyms_str
    return phrases_data

def translate_text_to_English(text: str, model_name: str):
    llm_client=initalize_local_ollama_environment(model_name)
    english_text=llm_client.query(text, "Role: You are a professional translator. Translate the text to English. Only do this, do not add any output that was not present in the input given above. Text translated to English:")
    return(english_text)


def clinical_notes_to_HPO(clinical_notes: str, model_name: str):
    #if check_and_initialize(): # Run your one-time initialization process here
    #    initialize_groq_environment()
    global llm_client
    llm_client=initalize_local_ollama_environment(model_name)
    try:
        start_time = time.time()  # Record the start time
        timestamped_print("Starting the HPO term extraction process.")

        df = pd.DataFrame({'clinical_note': [clinical_notes]})

        # Assign patient IDs if not provided
        df['patient_id'] = range(1, len(df) + 1)

        df = df.dropna(subset=['clinical_note'])  # Remove rows where 'clinical_note' is NaN
        df['clinical_note'] = df['clinical_note'].astype(str)  # Ensure 'clinical_note' column is of type string

        # Initialize empty DataFrames
        combined_results_df = pd.DataFrame()
        exact_matches_df = pd.DataFrame()
        non_exact_matches_df = pd.DataFrame()
        responses_df = pd.DataFrame()
        final_result_df = pd.DataFrame()

        # Initialize models and data
        timestamped_print("Initializing embeddings model")
        embeddings_model = initialize_embeddings_model()
        timestamped_print("Loading embedded documents")
        embedded_documents = load_embedded_documents(rag_hpo_dir + 'G2GHPO_metadata.npy')
        timestamped_print("Preparing embeddings list")
        embeddings_array = prepare_embeddings_list(embedded_documents)
        timestamped_print("Creating FAISS index")
        index = create_faiss_index(embeddings_array)

        # Process each remaining clinical note
        for _, row in df.iterrows():
            clinical_note = row['clinical_note']
            patient_id = row['patient_id']
            timestamped_print(f"Processing clinical note for patient_id {patient_id} ...") # modified for privacy
            result_df = process_row(clinical_note, system_message_I, embeddings_model, index, embedded_documents)
            if result_df is not None:
                result_df['patient_id'] = patient_id
                combined_results_df = pd.concat([combined_results_df, result_df], ignore_index=True)
        if combined_results_df.empty:
            timestamped_print("No data in combined_results_df. Exiting.")
            sys.exit(0)

        combined_results_df['phrase'] = combined_results_df['phrase'].str.lower()
        combined_results_df['unique_metadata'] = combined_results_df['unique_metadata'].apply(process_unique_metadata)

        #if 'process_exact_non_exact_matches' in steps_to_run:
        timestamped_print("Processing exact and non-exact matches.")

        # Add HPO terms for exact matches
        combined_results_df['HPO_Term'] = combined_results_df.apply(
            lambda row: extract_hpo_term(row['phrase'], row['unique_metadata']), axis=1
        )

        # Separate exact and non-exact matches
        exact_matches_df = combined_results_df.dropna(subset=['HPO_Term'])
        non_exact_matches_df = combined_results_df[combined_results_df['HPO_Term'].isna()]

        # Ensure 'HPO_Term' column exists and is of correct type
        if 'HPO_Term' in non_exact_matches_df.columns:
            # Identify indices where 'HPO_Term' is NaN (unprocessed entries)
            remaining_indices = non_exact_matches_df[non_exact_matches_df['HPO_Term'].isna()].index
            timestamped_print(f"Number of unprocessed non-exact matches: {len(remaining_indices)}")
        else:
            remaining_indices = non_exact_matches_df.index
            # Initialize 'HPO_Term' column with NaN values
            non_exact_matches_df['HPO_Term'] = np.nan
            timestamped_print(f"Total non-exact matches to process: {len(remaining_indices)}")

        # Check if non_exact_matches_df is empty
        if non_exact_matches_df.empty:
            timestamped_print("No non-exact matches found. Skipping HPO term generation for non-exact matches.")
            # Ensure 'HPO_Term' column exists
            if 'HPO_Term' not in non_exact_matches_df.columns:
                non_exact_matches_df['HPO_Term'] = pd.Series(dtype="object")
            # Remove 'generate_hpo_terms' from steps_to_run if present
            if 'generate_hpo_terms' in steps_to_run:
                steps_to_run.remove('generate_hpo_terms')

        #if 'generate_hpo_terms' in steps_to_run:
        timestamped_print("Generating HPO terms for non-exact matches.")
        timestamped_print(f"Processing {len(remaining_indices)} non-exact matches.")

        # Initialize responses DataFrame if empty
        if responses_df.empty:
            responses_df = pd.DataFrame(columns=['response'])

        # Process the remaining non-exact matches
        if len(remaining_indices) > 0:
            for idx in tqdm(remaining_indices, total=len(remaining_indices)):
                row = non_exact_matches_df.loc[idx]
                response_df = generate_hpo_terms(
                    pd.DataFrame([row]),
                    system_message_II
                )
                if response_df['response'].iloc[0] is not None:
                    responses_df = pd.concat([responses_df, response_df], ignore_index=True)
                    non_exact_matches_df.at[idx, 'HPO_Term'] = response_df['response'].iloc[0]
                else:
                    # Handle cases where response is None due to repeated failures
                    non_exact_matches_df.at[idx, 'HPO_Term'] = 'Error: Unable to process'
        else:
            timestamped_print("No unprocessed non-exact matches found.")
        #else:
        #    timestamped_print("All non-exact matches have been processed.")

        #if 'compile_final_results' in steps_to_run:
        timestamped_print("Compiling final results.")

        # Combine results and prepare final output
        final_combined_df = pd.concat([exact_matches_df, non_exact_matches_df], ignore_index=True)
        final_combined_df_grouped = final_combined_df.groupby('patient_id').apply(
            lambda group: group[['phrase', 'HPO_Term']].to_dict('records')
        )

        #print(final_combined_df_grouped) # commented out for privacy
        final_result_df = pd.DataFrame({
            'patient_id': final_combined_df_grouped.index,
            'HPO_Terms': final_combined_df_grouped.values
        })
        
        # Process the final results
        #process_results(final_result_df) 
        #print(final_result_df)
        new_data=process_hpo_simple(final_result_df)

        final_result_df_annotated = join_label_and_synonyms(new_data)

        timestamped_print(f"Total execution time: {time.time() - start_time:.2f} seconds")
        return(final_result_df_annotated)
        # Delete temporary files
        #for temp_file in temp_files:
        #    if os.path.exists(temp_file):
        #        os.remove(temp_file)

    except Exception as e:
        timestamped_print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
