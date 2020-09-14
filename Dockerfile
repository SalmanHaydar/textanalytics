FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR '/app'

COPY requirements1.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements1.txt

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5555"]

