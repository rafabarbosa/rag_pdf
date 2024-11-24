# Use the official Python image from the Docker Hub
FROM python:3.12.6

# Make a new directory to put our backend in.
RUN mkdir /backend

# Change the working directory.
WORKDIR /backend

# Copy only the requirements first
COPY requirements.txt /backend/
RUN pip install -r requirements.txt

# Install Tesseract OCR using apt-get
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    libgl1-mesa-glx \
    make \
    gcc \
    && python3 -m pip install -r requirements.txt \
    && apt-get remove -y --purge make gcc build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*


# Then copy the rest of the backend
COPY . /backend/

# Expose the port the app runs on
EXPOSE 8000
EXPOSE 11434

# Run the application in development:
# CMD uvicorn api:app --reload
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "api:app", "--reload"]
# CMD ["fastapi", "run", "app/main.py", "--port", "80"]