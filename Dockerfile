FROM jjanzic/docker-python3-opencv
WORKDIR /code
ADD flaskServer.py /code
ADD uploadedimages /code/uploadedimages
ADD yolov3.weights /code
ADD yolov3.cfg /code
ADD coco.names /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python","/code/flaskServer.py"]
#ENTRYPOINT ["python","/code/flaskServer.py"]

