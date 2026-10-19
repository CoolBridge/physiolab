FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m physiolab.synth
EXPOSE 8501
ENV PHYSIOLAB_DB=/data/physiolab.db
VOLUME /data
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]