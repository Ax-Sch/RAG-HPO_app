# Use the official Python slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install required Python packages # --no-cache-dir 
RUN pip install \
    contextlib2 \
    fastembed \
    fuzzywuzzy \
    langchain \
    langchain-core \
    langchain-groq \
    langchain_ollama \
    ratelimit \
    streamlit \
    langchain_community \
    tabulate \
    tqdm \
    faiss-cpu \
    numpy \
    pandas \
    requests \
    python-socketio

# initialize embeddings model to run without internet later on
RUN python3 -c 'from fastembed import TextEmbedding; embeddings_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5"); embeddings_model.embed("ABC")'

EXPOSE 8501
