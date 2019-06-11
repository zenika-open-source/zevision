FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="egor-jerome.akhanov@zenika.com"
WORKDIR /opt/face_recognition
COPY . .
RUN /bin/bash setup.sh
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["-m", "api.server"]
