FROM tensorflow/tensorflow

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install tensorflow_datasets && \
    mkdir Saved_Model

CMD ["python", "main.py"]
