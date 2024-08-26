FROM python:3.12-slim

COPY app/ /frontUI/app/
COPY images/ /frontUI/images/
COPY requirements.txt /frontUI/

WORKDIR /frontUI

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]