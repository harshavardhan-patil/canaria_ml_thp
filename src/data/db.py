import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from src.config import DATA_DIR
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import List, Tuple
import pandas as pd
import ast
import psycopg2.pool

load_dotenv()

# Get database connection parameters from environment variables
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
dbname = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

# Create a connection pool
min_conn = 5
max_conn = 20
connection_pool = None
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        min_conn, max_conn,
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
except:
    pass
finally:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        min_conn, max_conn,
        host="localhost",
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )

def get_connection():
    return connection_pool.getconn()

def release_connection(conn):
    connection_pool.putconn(conn)

def connect_to_db():
    """Establish connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
    except:
            conn = psycopg2.connect(
            host="localhost",
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
    return conn


def load_jobs_from_csv(file_path= DATA_DIR / 'jobs.csv'):
    """Load and preprocess jobs data from CSV file"""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} job records")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None



def insert_job_batch(cursor, job_data):
    """Insert a batch of job records into the jobs table. job_data should be list of tuples
    """
    try:
        execute_values(
            cursor,
            """
            INSERT INTO jobs 
            (lid, jobTitle, companyName, jobDescRaw, finalZipcode, finalState, 
            finalCity, companyBranchName, jobDescUrl, nlpBenefits, nlpSkills, 
            nlpSoftSkills, nlpDegreeLevel, nlpEmployment, nlpSeniority, 
            correctDate, scrapedLocation)
            VALUES %s
            ON CONFLICT (lid) DO UPDATE SET
                jobTitle = EXCLUDED.jobTitle,
                companyName = EXCLUDED.companyName,
                jobDescRaw = EXCLUDED.jobDescRaw,
                finalZipcode = EXCLUDED.finalZipcode,
                finalState = EXCLUDED.finalState,
                finalCity = EXCLUDED.finalCity,
                companyBranchName = EXCLUDED.companyBranchName,
                jobDescUrl = EXCLUDED.jobDescUrl,
                nlpBenefits = EXCLUDED.nlpBenefits,
                nlpSkills = EXCLUDED.nlpSkills,
                nlpSoftSkills = EXCLUDED.nlpSoftSkills,
                nlpDegreeLevel = EXCLUDED.nlpDegreeLevel,
                nlpEmployment = EXCLUDED.nlpEmployment,
                nlpSeniority = EXCLUDED.nlpSeniority,
                correctDate = EXCLUDED.correctDate,
                scrapedLocation = EXCLUDED.scrapedLocation
            """,
            job_data
        )
        return True
    except Exception as e:
        print(f"Error inserting batch data: {e}")
        return False

def insert_clean_job_batch(cursor, processed_data):
    """Insert a batch of job records into the jobs_processed. processed_data should be list of tuples
    """
    try:
        execute_values(
            cursor,
            """
                INSERT INTO jobs_processed (
                    lid, jobtitle, jobtitle_normalized, jobtitle_standardized, companyname, companyname_standardized,
                    finalcity, finalstate, finalzipcode, jobdesc_clean, jobdesc_lemmatized,
                    nlpskills, nlpsoftskills, nlpdegreelevel, nlpemployment, nlpseniority, correctdate
                ) VALUES %s
                ON CONFLICT (lid) DO UPDATE SET
                    jobtitle = EXCLUDED.jobtitle,
                    jobtitle_normalized = EXCLUDED.jobtitle_normalized,
                    jobtitle_standardized = EXCLUDED.jobtitle_standardized,
                    companyname = EXCLUDED.companyname,
                    companyname_standardized = EXCLUDED.companyname_standardized,
                    finalcity = EXCLUDED.finalcity,
                    finalstate = EXCLUDED.finalstate,
                    finalzipcode = EXCLUDED.finalzipcode,
                    jobdesc_clean = EXCLUDED.jobdesc_clean,
                    jobdesc_lemmatized = EXCLUDED.jobdesc_lemmatized,
                    nlpskills = EXCLUDED.nlpskills,
                    nlpsoftskills = EXCLUDED.nlpsoftskills,
                    nlpdegreelevel = EXCLUDED.nlpdegreelevel,
                    nlpemployment = EXCLUDED.nlpemployment,
                    nlpseniority = EXCLUDED.nlpseniority,
                    correctdate = EXCLUDED.correctdate
                """,
            processed_data
        )
        return True
    except Exception as e:
        print(f"Error inserting batch data in processed table: {e}")
        return False

def get_jobs() -> List[Tuple[str, str, str]]:
    """
    Get a batch of jobs from the database
    
    Returns: List of tuples containing (lid, job_title, job_description)
    """
    conn = None
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                lid, 
                jobtitle, 
                companyname,
                finalcity,
                finalstate,
                finalzipcode,
                jobdesc_clean
            FROM jobs_processed
        """
        
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        print(f"Error getting jobs batch: {e}")
        raise
    finally:
        if conn:
            conn.close()



def process_list_column(value):
    """Convert string representation of a list to an actual list."""
    if pd.isna(value) or value == '[]':
        return []
    
    try:
        # Handle different formats that might appear in the CSV
        if isinstance(value, str):
            # Try to safely evaluate the string as a Python literal
            return ast.literal_eval(value)
        elif isinstance(value, list):
            return value
        else:
            return []
    except (SyntaxError, ValueError):
        # If evaluation fails, handle it as a comma-separated string
        return [item.strip() for item in value.strip('[]').split(',') if item.strip()]
