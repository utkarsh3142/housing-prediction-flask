FROM python 3.7.5

COPY model /app/model
COPY app.py /app
COPY utils.py /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]