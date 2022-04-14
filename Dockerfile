FROM tensorflow/tensorflow

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install tensorflow_datasets

CMD ["python", "main.py"]
