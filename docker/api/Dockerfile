# Use the official Python slim image
FROM python:3.11-slim

# Set environment variables for non-interactive installations
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies
RUN pip install requests flask #--no-cache-dir 

EXPOSE 5000
