# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure stdout/stderr are unbuffered so logs work properly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed by some ML packages like OpenCV if added later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirement.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the current directory contents into the container at /app
# This includes the model file my_model.h5
COPY . /app/

# Create an unprivileged user to run the application
# Hugging Face Spaces often run as user 1000
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change working directory to the user's home directory and copy files there
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Reassemble the split TFLite model chunks into the full model file
RUN cat model.tflite.part* > model.tflite && rm model.tflite.part*

# Create upload directory and set permissions
RUN mkdir -p $HOME/app/static/uploads && chmod -R 777 $HOME/app/static/uploads

# Expose port 7860 which is used by Hugging Face Spaces
EXPOSE 7860

# Run the application using gunicorn for production grade deployment
# Binds to the $PORT environment variable provided by Render, or defaults to 7860
CMD gunicorn -b 0.0.0.0:${PORT:-7860} --timeout 120 --workers 2 app:app
