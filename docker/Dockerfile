# #FROM ubuntu:latest
# FROM python:3.8.1
# #COPY ./*.py /contapi/
# COPY ./api/app.py /contapi/api/app.py
# COPY ./requirements.txt /contapi/requirements.txt
# COPY ./svm_gamma=0.0005_C=2.joblib /contapi/svm_gamma=0.0005_C=2.joblib
# RUN pip3 install --no-cache-dir -r /contapi/requirements.txt
# WORKDIR /contapi
# EXPOSE 5000
# ENV FLASK_APP="api/app.py"
# CMD ["flask", "run", "--host", "0.0.0.0"]


# FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /expf/
COPY ./requirements.txt /expf/requirements.txt
COPY ./entrypoint.sh / expf/entrypoint.sh
ENTRYPOINT ["expf/entrypoint.sh"]
RUN pip3 install --no-cache-dir -r /expf/requirements.txt
WORKDIR /expf

CMD ["python3", "./plots_graph.py",]