FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-get update
RUN apt-get install -y protobuf-compiler ffmpeg libsm6 libxext6

RUN mkdir decap
WORKDIR /decap
ADD ./* .
# upgrade pip and install requirements
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /
# get latest tensorflow models and compile object-detection for python package
RUN clone https://github.com/tensorflow/models.git
RUN cd models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN python -m pip install .

CMD []