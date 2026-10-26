#!/bin/sh
# Runs once, the first time the postgres volume is created.
# The app database ("chaincommand") is created by the image itself; MLflow gets its own database.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-SQL
    CREATE DATABASE mlflow OWNER "$POSTGRES_USER";
SQL
