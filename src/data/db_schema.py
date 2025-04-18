import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from src.data.db import connect_to_db

load_dotenv()

# Get database connection parameters from environment variables
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
dbname = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

def setup_db_schema():
    """ One-time setup of Postgres """
    # Connect to PostgreSQL
    conn = connect_to_db()
    conn.autocommit = True
    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{dbname}'")
    if not cursor.fetchone():
        cursor.execute(f"CREATE DATABASE {dbname}")

    # Connect to the newly created database
    conn.close()
    conn = connect_to_db()
    cursor = conn.cursor()

    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        lid VARCHAR(255) PRIMARY KEY,
        jobTitle TEXT,
        companyName TEXT,
        jobDescRaw TEXT,
        finalZipcode VARCHAR(20),
        finalState TEXT,
        finalCity TEXT,
        companyBranchName TEXT,
        jobDescUrl TEXT,
        nlpBenefits TEXT[],
        nlpSkills TEXT[],
        nlpSoftSkills TEXT[],
        nlpDegreeLevel TEXT[],
        nlpEmployment TEXT,
        nlpSeniority TEXT,
        correctDate TIMESTAMP,
        scrapedLocation TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs_processed (
        lid VARCHAR(255) PRIMARY KEY,
        jobtitle TEXT,
        jobtitle_normalized TEXT,
        jobtitle_standardized TEXT,
        companyname TEXT,
        companyname_standardized TEXT,
        finalcity TEXT,
        finalstate TEXT,
        finalzipcode VARCHAR(20),
        jobdesc_clean TEXT,
        jobdesc_lemmatized TEXT,
        nlpskills TEXT[],
        nlpsoftskills TEXT[],
        nlpdegreelevel TEXT[],
        nlpemployment TEXT,
        nlpseniority TEXT,
        correctdate TIMESTAMP
    )
    """)

    # all-MiniLM-L6-v2 -> 384 emb dim
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_embeddings (
        lid VARCHAR(255) PRIMARY KEY REFERENCES jobs(lid),
        embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_duplicates (
        lid1 VARCHAR(255) REFERENCES jobs(lid),
        lid2 VARCHAR(255) REFERENCES jobs(lid),
        similarity_score FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (lid1, lid2)
    )
    """)

    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lid ON jobs(lid)")

    conn.commit()
    conn.close()

    print("Database setup completed successfully.")
