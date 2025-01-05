from flask import Flask, request, jsonify
import requests
from retrieve_genes_external import get_genes  # Import the function


app = Flask(__name__)

@app.route("/get_genes", methods=["GET"])
def get_genes_api():
    # Get the HPO term from the query parameter
    hpo_term = request.args.get("hpo_term")
    if not hpo_term:
        return jsonify({"error": "Missing HPO term"}), 400

    # Fetch gene names using the get_genes function
    genes = get_genes(hpo_term)
    if genes is not None:
        return jsonify({"hpo_term": hpo_term, "genes": genes})
    else:
        return jsonify({"error": "Failed to retrieve data"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
