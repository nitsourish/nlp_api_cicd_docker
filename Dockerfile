FROM python:3.8.8
COPY requirements.txt /app/requirements.txt
RUN cd /app && \
	pip install -r requirements.txt
ADD . /app
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["app.py"]
