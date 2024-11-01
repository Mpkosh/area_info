FROM python:3.11-slim

COPY requirements.txt requirements.txt


RUN pip install -r requirements.txt
RUN pip install gunicorn pymysql cryptography

COPY app_package app_package
COPY migrations migrations
COPY app_plots.py config_f.py boot.sh ./
RUN chmod a+x boot.sh

ENV FLASK_APP app_plots.py
ENV SOCIAL_API "http://10.32.1.108:8000/"
ENV TERRITORY_API "http://10.32.1.107:5300/"


EXPOSE 5000
ENTRYPOINT ["./boot.sh"]