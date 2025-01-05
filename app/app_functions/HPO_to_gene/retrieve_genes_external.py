import requests

def get_genes(hpo_term):
    url = f"https://ontology.jax.org/api/network/annotation/{hpo_term}"
    # Make the API request
    response = requests.get(url, headers={"accept": "application/json"})
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  
        # Extract the gene names
        genes = [gene["name"] for gene in data.get("genes", [])]
        print("Gene names:", genes)
        return(genes)
    else:
        print(f"Failed to retrieve data: {response.status_code}")