import sys
import os
import streamlit as st
import pandas as pd
import requests
import re

from app_functions.RAG_HPO.RAG_HPO import clinical_notes_to_HPO, translate_text_to_English
from app_functions.llm_connection.llm import get_avail_models



# Streamlit app starts here
st.title("Clinical Notes to HPO Terms")

model_name = st.selectbox(
        "Choose a Model:",
        options=get_avail_models(),
        index=0  # Default model
    )

if "clinical_notes" not in st.session_state:
    st.session_state.clinical_notes=""


# Input box for clinical notes (user enters clinical notes as a string)
clinical_notes = st.text_area(label="Enter clinical notes:", value=st.session_state.clinical_notes, height=200)


# If the user wants to translate the input text to English
if st.button("Optional: Translate to English") and clinical_notes:
    translated_notes=translate_text_to_English(clinical_notes, model_name)
    st.session_state.clinical_notes=translated_notes
    st.rerun()

def annotate_clinical_notes(_clinical_notes, _hpo_df):
    # loop over rows of pandas df _hpo_df
    # for each row search for _hpo_df["Phenotype name"] in the string _clinical notes (ignore upper/lower case) and replace hits with 
    # red[**hit**]
    _hpo_df=_hpo_df.drop_duplicates()
    for _, row in _hpo_df.iterrows():
        phenotype = row["Phenotype name"]
        hpo_label = row["label"]
        # Use regex to find case-insensitive matches and replace them
        _clinical_notes = re.sub(
            rf"(?i)\b{re.escape(phenotype)}\b",  # Regex pattern for case-insensitive whole word match
            f":red[**{phenotype} (->{hpo_label})**]",
            _clinical_notes
        )
    return _clinical_notes



# Button to trigger HPO term extraction
if st.button("Get HPO Terms"):
    hpo_df = clinical_notes_to_HPO(clinical_notes, model_name)
    # annotate clinical notes
    clinical_notes_annotated=annotate_clinical_notes(clinical_notes, hpo_df)
    if "source_texts" not in st.session_state:
        st.session_state.source_texts = [clinical_notes_annotated]
    else:
        st.session_state.source_texts.append(clinical_notes_annotated)
    
    if "hpo_df" not in st.session_state:
        st.session_state.hpo_df = hpo_df
    else:
        st.session_state.hpo_df = pd.concat([hpo_df, st.session_state.hpo_df], ignore_index=True)


# If the DataFrame is available in session state, display it and create checkboxes
if 'hpo_df' in st.session_state:
    hpo_df = st.session_state.hpo_df
    hpo_df = hpo_df.drop_duplicates()
       
    # show history
    if "source_texts" in st.session_state:
        for source_text in st.session_state.source_texts:
            st.write(source_text)
    
    # Display the DataFrame in the Streamlit app
    st.write("Extracted HPO Terms:", hpo_df)

    # Initialize a list to keep track of selected rows
    selected_rows = []

    # Create checkboxes for each row
    for index, row in hpo_df.iterrows():
        checkbox = st.checkbox(f"Select HPO Term {row['HPO_ID']} - {row['label']}", key=index)
        if checkbox:
            selected_rows.append(row)

    if selected_rows:
        selected_rows_df = pd.DataFrame(selected_rows)
        selected_rows_sorted = selected_rows_df.sort_values(by='label')
        # Export the selected rows' HPO IDs
        selected_hpo_ids = "; ".join(selected_rows_sorted['HPO_ID'])
        
        st.write("**Selected HPO IDs:**")
        st.write(selected_hpo_ids)

        selected_hpo_ids_labels = "; ".join(selected_rows_sorted.apply(lambda row: f"{row['HPO_ID']} - {row['label']}", axis=1))
        st.write("**Selected HPO IDs and labels alphabetically sorted, semicolon separated:**")
        st.write(selected_hpo_ids_labels)
        
        st.write("**Selected HPO IDs and labels alphabetically sorted, newline separated:**")
        for index, row in selected_rows_sorted.iterrows():
            st.write(f"{row['HPO_ID']} - {row['label']}", key=index)


def get_genes_from_api(hpo_term):
    api_url = "http://retrieve_HPO_api:5000/get_genes"
    # Make the GET request
    response = requests.get(api_url, params={"hpo_term": hpo_term})
   
    # Check the response
    if response.status_code == 200:
        data = response.json()
        print("HPO Term:", data.get("hpo_term"))
        print("Genes:", data.get("genes"))
        return(data.get("genes"))
    else:
        print("Error:", response.json().get("error"))
        return("Error")


if st.button("Get Genes"):
    genes_df=pd.DataFrame()
    for index, row in selected_rows_sorted.iterrows():
        genes = get_genes_from_api(row['HPO_ID'])
        if not genes: 
            continue
        temp_genes_df = pd.DataFrame({
            "HPO Term": [row['HPO_ID']] * len(genes),
            "HPO Label": [row['label']] * len(genes),
            "Gene Name": genes
        })
        genes_df=pd.concat([genes_df, temp_genes_df], ignore_index=True)
    st.write(genes_df)

if st.button("Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


st.write("Note: Using the 'Get Genes' button will transmit the selected HPO terms to https://ontology.jax.org and retrieve the corresponding genes from this server. All other data will remain on the local server.")
