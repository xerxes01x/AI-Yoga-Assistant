FROM python:3.9-slim

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install system deps required by OpenCV/TensorFlow runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install the python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# run with gunicorn in container
ENV PORT=8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]