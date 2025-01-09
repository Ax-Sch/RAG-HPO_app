import sys
import os
import streamlit as st
import pandas as pd
import requests

from app_functions.RAG_HPO.RAG_HPO import clinical_notes_to_HPO, translate_text_to_English
from app_functions.llm_connection.llm import get_avail_models



# Streamlit app starts here
st.title("Clinical Notes to HPO Terms")

model_name = st.selectbox(
        "Choose a Model:",
        options=get_avail_models(),
        index=0  # Default model
    )

# Input box for clinical notes (user enters clinical notes as a string)
clinical_notes = st.text_area("Enter clinical notes:", height=200)


# If the user wants to translate the input text to English
if st.button("Optional: Translate to English") and clinical_notes:
    translated_notes=translate_text_to_English(clinical_notes, model_name)
    st.session_state.translated_notes=translated_notes

# show translation
if "translated_notes" in st.session_state:
    st.write(f":red[**Translated Clinical notes:**] {st.session_state.translated_notes}")

# Button to trigger HPO term extraction
if st.button("Get HPO Terms"):
    if clinical_notes:
        if "translated_notes" in st.session_state:
            hpo_df = clinical_notes_to_HPO(st.session_state.translated_notes, model_name)
        else:
            hpo_df = clinical_notes_to_HPO(clinical_notes, model_name)
        # Store the DataFrame in session state so it persists across reruns
        st.session_state.hpo_df = hpo_df
    else:
        st.warning("Please enter some clinical notes.")

# If the DataFrame is available in session state, display it and create checkboxes
if 'hpo_df' in st.session_state:
    hpo_df = st.session_state.hpo_df
    
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
        # Export the selected rows' HPO IDs
        selected_hpo_ids = "; ".join([row['HPO_ID'] for row in selected_rows])
        st.write("Selected HPO IDs:")
        st.write(selected_hpo_ids)


        selected_hpo_ids_labels = "; ".join([row['HPO_ID'] + " - "+ row['label'] for row in selected_rows])
        st.write("Selected HPO IDs and labels:")
        st.write(selected_hpo_ids_labels)


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
    for row in selected_rows:
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


st.write("Note: Using the "Get Genes" button will transmit the selected HPO terms to https://ontology.jax.org and retrieve the corresponding genes from this server. All other data will remain on the local server.")
