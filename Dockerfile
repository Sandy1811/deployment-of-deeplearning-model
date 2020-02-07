#User Python as base Image
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

#Installing required packages
RUN pip install -r requirements.txt
# while read requirement; do conda install --yes $requirement;or pip install $requirement; done < requirements.txt \
