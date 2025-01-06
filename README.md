# RAG-HPO App

This repository contains an implementation of [RAG-HPO](https://github.com/PoseyPod/RAG-HPO) and a simple chat app that can run locally and offline to ensure privacy. RAG-HPO was adopted to extract HPO terms from medical notes of individual cases. Additionally, genes, linked to HPO-terms can be retrieved. The chat app allows to create custom prompts for the local large language model (LLM). The frontend is built using **Streamlit**, and **Ollama** is used for executing LLMs. This comes without any warranty (see licence). 

---

## Table of Contents
1. [Idea](#idea)
2. [Installation](#installation)
   - [Requirements](#requirements)
   - [Download LLM Files](#download-llm-files)
   - [Initialize HPO Vector Database](#initialize-hpo-vector-database)
3. [Run](#run)
4. [Examples](#examples)
5. [Services](#services)
6. [Remarks](#remarks)

---

## Idea

This application is designed to run in isolated containers to ensure security:
- All containers use an **internal network** without internet access for communication amongst each other.
- **Streamlit** and **Ollama** containers operate without internet and write access to the local file system.
- Communication with external networks or local file storage happens via dedicated containers.

---

## Installation

### Requirements
- **Docker** (tested version: 27.3.1) with **Docker Compose** (v2.29.7)

Clone the repository into the desired directory you would like to place it:
```bash
git clone https://github.com/Ax-Sch/RAG-HPO_app
cd RAG-HPO_app
```

### Download LLM Files
You can replace llama3.1:8b by your prefered LLM (see ollama website) or add additional LLMs.
#### Option 1: Ollama in a Docker Container
```bash
# Run the Ollama Docker container and store models in the llm_files directory:
docker run -d -v ./llm_files:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Pull the required model (e.g., llama3.1:8b):
ollama pull llama3.1:8b

# Stop and remove the Ollama container:
docker stop ollama
docker rm ollama
```

#### Option 2: Local Ollama Installation
```bash
# Pull the required model:
ollama pull llama3.1:8b
```

### Initialize HPO Vector Database
The download link to the HPO json file is given as argument to the init.py script, and can be adopted, if needed:
```bash
docker build -f ./docker/app/Dockerfile -t initialize_vector ./docker/
docker run -v "./app:/app" initialize_vector python3 app/app_functions/RAG_HPO/init.py \
   https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-12-12/hp.json
docker rm initialize_vector
```

---

## Run

### Syntax
```bash
docker-compose [profiles] up --build [add "-d" to run in the background]
```

### Two profiles must be set:
1. **Choose how ollama should be run**:
   - **CPU Only:** `--profile ollama_container`
   - **GPU:** `--profile ollama_container_gpu`  
     *Settings are for a NVIDIA GPU. Refer to [Ollama Docker Help](https://hub.docker.com/r/ollama/ollama) for troubleshooting.*
   - **Mac (Local Installation):** `--profile ollama_proxy`

2. **Choose how to serve the streamlit app**:
   - **Localhost:** `--profile serve_local`  
     *App available at `127.0.0.1:8501` via a web browser.*
   - **SSH Server:** `--profile serve_ssh`  
     *A SSH server will allow forwarding the app within the local network (see Examples below).*  
     Add users by editing:  
     `RAG-HPO_app/docker/ssh_server/users/users.txt`
     The format is: USERNAME1:PASSWORD1,USERNAME2:PASSWORD2,...

---

## Examples

### Ollama on CPU and local access only:
```bash
docker-compose --profile ollama_container --profile serve_local up --build
```
*Access the app via browser at `127.0.0.1:8501`.*

### Ollama on GPU and SSH Server:
1. **Run on Server**:
   ```bash
   docker-compose --profile ollama_container_gpu --profile serve_ssh up --build
   ```
2. **Access from Client**:
   ```bash
   # adjust test1 to one of the usernames you provided via the users.txt file (see above)
   ssh -L 8501:app:8501 -p 2222 test1@[SERVER_IP]
   ```
   *Access the app via browser at `127.0.0.1:8501`.*

---

## Services

### Ollama Services
- **`ollama`**: Ollama container, CPU only (internal network only, read access to model files only).  
- **`ollama_gpu`**: GPU-enabled ollama server (internal network only, read access to model files only).  
- **`ollama_proxy`**: Nginx proxy for Ollama (internal and external networks).  

### Applications
- **`app`**: Streamlit app (internal network only, read-only access to app files).  

### APIs
- **`retrieve_hpo_api`**: API for HPO data retrieval (internal and external networks, no access to local files).  
- **`chat_app_file_mod_api`**: API for prompt management (internal network only, read/write access to local files).  

### Proxies for Streamlit App
- **`reverse_proxy_streamlit`**: Nginx proxy for serving the Streamlit app at 127.0.0.1 (internal and external networks).  
- **`ssh-server`**: SSH server for port forwarding only (internal and external networks).  

---

## Remarks

For feedback, contact: **axel.schmidt@ukbonn.de**

### Troubleshooting
#### Error: `Error response from daemon: network XXX not found`
Run the following command:
```bash
docker-compose [profiles] up --force-recreate
```