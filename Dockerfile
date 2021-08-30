FROM tensorflow/tensorflow:2.6.0-gpu

RUN mkdir decap
WORKDIR /decap
ADD ./* .
# upgrade pip and install requirements
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD []