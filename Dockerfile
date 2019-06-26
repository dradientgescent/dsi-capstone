FROM python:3.6.8-slim

RUN apt-get update

# Set the working directory
WORKDIR /eyenet

# Install libraries
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy files to container
COPY ./src ./src
