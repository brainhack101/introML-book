FROM jupyter/scipy-notebook

LABEL maintainer="Pierre Bellec <pierre.bellec@gmail.com>"

USER jovyan

# Copying the repository inside the container
COPY . /home/jovyan

# Installing Kamalaker's MAIN fetcher
RUN pip install -r requirements.txt

# Downloading the data
RUN ["/bin/bash", "/home/jovyan/data_fetch.sh"]
