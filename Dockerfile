# Use official Python 3.12 base image
FROM python:3.12.4

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Streamlit
EXPOSE 8080

# Start the Streamlit app
ENTRYPOINT [ "streamlit", "run", "digit1.py", "--server.port=8080", "--server.address=0.0.0.0" ]
