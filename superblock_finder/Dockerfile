FROM    python:3.10

ADD     requirements.txt /requirements.txt
RUN     pip install --upgrade pip && pip install -r /requirements.txt

RUN     mkdir -p /data/tmp/_scrap && mkdir -p /data/cities
ADD     superblocks /superblocks
ADD     preprocess_osm.py /preprocess_osm.py
ADD     superblock.py /superblock.py
ADD     deu_pd_2020_1km.tif /deu_pd_2020_1km.tif
ADD     _types /_types

ADD     app.py /app.py

EXPOSE  8000
CMD     ["uvicorn", "app:app", "--host='0.0.0.0'", "--port=8000"]