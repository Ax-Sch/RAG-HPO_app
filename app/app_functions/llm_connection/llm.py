from langchain.chat_models import ChatOllama
import socket
#import time
import requests
from requests.exceptions import HTTPError
from fastembed import TextEmbedding
#from langchain_ollama import OllamaEmbeddings


# Function to get base URL for Ollama
def get_base_url():
    host_ip = socket.gethostbyname("ollama")
    base_url = "http://" + str(host_ip) + ":11434"
    print("base_url: " + base_url)
    return base_url

# Initialize the Ollama chat model
def get_llm(model_name):
    llm = ChatOllama(
        model=model_name,  # Replace with the correct model name if needed
        temperature=0,
        base_url=get_base_url()
    )
    print("Initialized LLM")
    return(llm)


####### HPO RAG:
class LLMClient:
    """Class to manage LLM API configurations and queries."""
    def __init__(self, api_key, base_url, model_name="llama3-groq-70b-8192-tool-use-preview", 
                 max_tokens_per_day=500000, max_queries_per_minute=30, temperature=0.0):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.total_tokens_used = 0
        self.temperature = temperature
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def query(self, user_input, system_message):
        """Sends a query to the LLM API."""
        print(f"TOTAL_TOKENS_USED before query: {self.total_tokens_used}")

        # Check token limit
        estimated_tokens = len(user_input.split()) + len(system_message.split())
        if self.total_tokens_used + estimated_tokens > self.max_tokens_per_day:
            raise Exception("Token limit exceeded for the day.")

        # Enforce rate limit
        #time.sleep(60 / self.max_queries_per_minute)

        # Construct the payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            "temperature": self.temperature,  # Use the temperature from the instance
        }

        # Send the request
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        # Update token usage
        self.total_tokens_used += estimated_tokens

        # Parse and return the response content
        result = response.json()
        return result["choices"][0]["message"]["content"] if "choices" in result else "No content returned."



def initalize_local_ollama_environment(model_name="llama3.1"):
    """Initializes the LLMClient with user input or default settings."""
    #global INITIALIZED, llm_client
    try:
        host_ip = get_base_url()
        print(host_ip)
    except socket.gaierror as e:
        print("error getting IP")# Gather inputs from the user
    print("Starting Ollama connection")
    llm_client = LLMClient(api_key="ollama", 
                           base_url=str(host_ip)+"/v1/chat/completions", 
                           model_name= model_name, max_tokens_per_day=99999999, max_queries_per_minute=100)
    return(llm_client)


# Embeddings related functions
def initialize_embeddings_model():
    """Initializes the embeddings model for processing clinical notes."""
    model_name = "BAAI/bge-small-en-v1.5"
    try:
        embeddings_model = TextEmbedding(model_name=model_name, local_files_only=True)
        return embeddings_model
    except Exception:
        exit(1)  # Exit on failure to initialize the embeddings model
        print("Error: Unable to initialize the embeddings model.")

def get_avail_models():
    url = "http://ollama:11434/api/tags"
    # Make the API request
    response = requests.get(url, headers={"accept": "application/json"})
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  
        # Extract the gene names
        model_names = [model["name"] for model in data.get("models", [])]
        print("Model names:", model_names)
        return(model_names)
    else:
        print(f"Failed to retrieve data: {response.status_code}")
