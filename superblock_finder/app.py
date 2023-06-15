import json

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from psycopg2 import connect
from os import environ

from _types import Region, Coord
from superblock import find_superblocks

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

            try:
                job_id = result[0][0]
                con.commit()
            except Exception:
                return HTTPException(status_code=500, detail="Job not added successfully to database.")

            gdf_street_areas_all, intersect_buildings, all_blocks, blocks_no_street_all = find_superblocks(job_id=job_id, region_of_interest=roi)

            try:
                intersect_buildings.drop(columns=["created_at"], inplace=True)
            except:
                pass

            gdf_street_areas_all.to_crs(epsg=4326, inplace=True)
            intersect_buildings.to_crs(epsg=4326, inplace=True)
            all_blocks.to_crs(epsg=4326, inplace=True)
            blocks_no_street_all.to_crs(epsg=4326, inplace=True)

            streets_geojson = gdf_street_areas_all.to_json()
            buildings_geojson = intersect_buildings.to_json()
            blocks_geojson = all_blocks.to_json()
            blocks_no_street_geojson = blocks_no_street_all.to_json()

            return dict(
                job_id=job_id,
                streets=json.loads(streets_geojson),
                buildings=json.loads(buildings_geojson),
                blocks=json.loads(blocks_geojson),
                blocks_no_street=json.loads(blocks_no_street_geojson)
            )
