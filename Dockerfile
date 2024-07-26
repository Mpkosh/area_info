FROM python:3.11-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn pymysql cryptography

COPY app_package app_package
COPY migrations migrations
COPY app_plots.py config_f.py boot.sh ./
RUN chmod a+x boot.sh

ENV FLASK_APP app_plots.py

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]