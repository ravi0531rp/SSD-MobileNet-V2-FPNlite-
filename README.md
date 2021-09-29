# SSD-MobileNet-V2-FPNlite-

This repository contains an implementation of the Tensorflow Object Detection API based Transfer Learning on SSD MobileNet V2 FPNLite Architecture. You can use the steps mentioned below to do transfer learning on any other model present in the Model Zoo of Tensorflow.

The implementation involves using <a href ="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md">Tensorflow Object Detection API</a>

## PLEASE MAKE SURE THAT YOU PERFORM ALL THE INSTRUCTIONS IN A VIRTUAL ENVIROMENT (PREFEREABLY ON CLOUD; PAPERSPACE OR COLAB IS FINE AS WELL)!!!!

## Windows Implementation

* Step 1 : Download and Install anaconda. Please do this to avoid trouble of installing specific packages manually.
* Step 2 : conda create -n YouEnvName python=3.6 anaconda
* Step 3 : git clone https://github.com/tensorflow/models.git
* Step 4 : cd models/research
* Step 5 : Here we need to build protos, so, download protoc.exe from web and use PATH/TO/PROTOC/protoc.exe object_detection/protos/*.proto --python_out=.
* Step 6 : Copy the setup.py file from object_detection/packages/tf2/ to models/research
* Step 7 : python -m pip install . # don't use that command what's mentioned in the documentation , use this one to avoid errors
* Step 8 : It might have already installed Tensorflow or do pip install tensorflow==2.6.0 or pip install tensorflow-gpu==2.6.0
* Step 8 : Now we need to generate data. Pick any number of classes for the Model to train on. 
* Step 9 : Oh Wait!! Where's the model???
* Step 10 : Go to  <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">tensorflow 2 Model zoo </a>
* Step 11 : You can pick any Model from there as long as it does Object Detection. I chose <a href="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz">SSD MobileNet V2 FpnLite </a>
* Step 12 : Download and extract the files. You will find(since we are dealing in TF2 and not TF1) a) checkpoint folder 2) saved_model folder 3) pipeline.config
* More incoming.............


## Linux Implementation

* Step 1 : Download and Install anaconda. Please do this to avoid trouble of installing specific packages manually.
* Step 2 : conda create -n YouEnvName python=3.6 anaconda
* Step 3 : git clone https://github.com/tensorflow/models.git
* Step 4 : cd models/research
* Step 5 : Here we need to build protos, protoc object_detection/protos/*.proto --python_out=.
* Step 6 : Copy the setup.py file from object_detection/packages/tf2/ to models/research -->> cp object_detection/packages/tf2/setup.py .
* Step 7 : python -m pip install . # don't use that command what's mentioned in the documentation , use this one to avoid errors
* Step 8 : It might have already installed Tensorflow or do pip install tensorflow==2.6.0 or pip install tensorflow-gpu==2.6.0
* Step 8 : Now we need to generate data. Pick any number of classes for the Model to train on. 
* Step 9 : Oh Wait!! Where's the model???
* Step 10 : Go to  <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">tensorflow 2 Model zoo </a>
* Step 11 : You can pick any Model from there as long as it does Object Detection. I chose <a href="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz">SSD MobileNet V2 FpnLite </a>
* Step 12 : Download and extract the files. You will find(since we are dealing in TF2 and not TF1) a) checkpoint folder 2) saved_model folder 3) pipeline.config
* More incoming.............
