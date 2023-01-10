from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from psycopg import connect
from os import environ
from _types import Region, Coord

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/region")
def add_job(roi: Region):
    with connect(dbname=environ["POSTGRES_DATABASE"], host=environ["POSTGRES_HOST"], user=environ["POSTGRES_USER"],
                 password=environ["POSTGRES_PASSWORD"]) as con:
        with con.cursor() as cur:
            sql = "CREATE TABLE IF NOT EXISTS jobs (" \
                  " id serial NOT NULL PRIMARY KEY," \
                  " roi geometry NOT NULL," \
                  " name text, " \
                  " created_at timestamp WITHOUT TIME ZONE default current_timestamp" \
                  ");"
            cur.execute(sql)
            con.commit()

            if len(roi.coords) < 3:
                return HTTPException(status_code=500, detail="No valid polygon given. You need to provide at least 3 points.")

            coords = roi.coords + [roi.coords[0]]
            polygon = "POLYGON((" + ",".join([f"{point.lng} {point.lat}" for point in coords]) + "))"

            sql = f"INSERT INTO jobs (name, roi) " \
                  f"VALUES (" \
                  f"    '{roi.name}', " \
                  f"    ST_SetSRID(ST_GeomFromText('{polygon}'), 4326)" \
                  f")" \
                  f"RETURNING *"
            cur.execute(sql)
            result = cur.fetchall()

    return {"coords": roi.coords, "name": roi.name, "return": result}
