#User Python as base Image
FROM kaixhin/cuda-theano:7.5
FROM continuumio/miniconda3:4.6.14

MAINTAINER sandeep <sndp1811@gmail.com>

# Copy all the content of Current directory to /app
ADD . /app

#Use working directory /app
WORKDIR /app

#Run Python program
CMD ["python", "app.py"]

#Open port 5000
EXPOSE 5000

#Set Environment Variable
ENV name base1

ARG python_version=3.6

#Installing required packages
RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    while read requirement; do conda install --yes $requirement; done < requirements.txt
    pip install Keras==2.3.1
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete


#User Python as base Image
# FROM kaixhin/cuda-theano:7.5
# FROM continuumio/miniconda3:4.6.14
# FROM ubuntu:latest

MAINTAINER sandeep <sndp1811@gmail.com>

# Install latest updates
# RUN apt-get update -y

# Install Python and build libraries
# RUN apt-get install -y python-pip python-dev build-essential

FROM debian:buster-slim
# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates \
		netbase \
	&& rm -rf /var/lib/apt/lists/*


# Copy all the content of Current directory to /app
ADD . /app

#Use working directory /app
WORKDIR /app

#Run Python program
CMD ["python", "app.py"]

#Open port 5000
EXPOSE 5000

#Set Environment Variable
ENV name base1

ARG python_version=3.6

#Installing required packages
RUN pip install -r requirements.txt
# while read requirement; do conda install --yes $requirement;or pip install $requirement; done < requirements.txt \
