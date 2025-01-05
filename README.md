# RAG-HPO App

This repository contains an implementation of [RAG-HPO](https://github.com/PoseyPod/RAG-HPO) and a chat app that can run locally and offline to ensure privacy. The frontend is built using **Streamlit**, and **Ollama** is used for executing large language models (LLMs).

---

## Table of Contents
1. [Idea](#idea)
2. [Installation](#installation)
   - [Requirements](#requirements)
   - [Download LLM Files](#download-llm-files)
   - [Initialize HPO Vector Database](#initialize-hpo-vector-database)
3. [Run](#run)
4. [Examples](#examples)
5. [Remarks](#remarks)

---

## Idea

This application is designed to run in isolated containers to ensure security:
- All containers use an **internal network** without internet access for communication.
- **Streamlit** and **Ollama** containers operate without internet or write access to the local file system.
- Communication with external networks or local file storage happens via dedicated containers.

---

## Installation

### Requirements
- **Docker** (tested version: 27.3.1) with **Docker Compose** (v2.29.7)

Ensure the working directory is the local repository's directory.

### Download LLM Files
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
```bash
docker build -f ./docker/app/Dockerfile -t initialize_vector ./docker/
docker run -v "./app:/app" initialize_vector /bin/bash app/app_functions/RAG_HPO/initialize.sh
docker rm initialize_vector
```

---

## Run

### Syntax
```bash
docker-compose [profiles] up --build
```

### Profiles
1. **LLM Profiles**:
   - **CPU Only:** `--profile ollama_container`
   - **GPU:** `--profile ollama_container_gpu`  
     *Settings are for a NVIDIA GPU. Refer to [Ollama Docker Help](https://hub.docker.com/r/ollama/ollama) for troubleshooting.*
   - **Mac (Local Installation):** `--profile ollama_proxy`

2. **Streamlit Profiles**:
   - **Localhost:** `--profile serve_local`  
     *App available at `127.0.0.1:8501`.*
   - **SSH Server:** `--profile serve_ssh`  
     *A SSH server will allow forwarding the app within the local network.*  
     Add users by editing:  
     `RAG-HPO_app/docker/ssh_server/users/users.txt`
     The format is: USERNAME1:PASSWORD1,USERNAME2:PASSWORD2,...

---

## Examples

### Ollama on CPU and Local Access:
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
   ssh -L 8501:app:8501 -p 2222 test1@[SERVER_IP]
   ```
   *Access the app via browser at `127.0.0.1:8501`.*

---

## Remarks

For feedback, contact: **axel.schmidt@ukbonn.de**

### Troubleshooting
#### Error: `Error response from daemon: network XXX not found`
Run the following command:
```bash
docker-compose [profiles] up --force-recreate
```
