from src.data.db_schema import setup_db_schema
from src.config import DATA_DIR
import pandas as pd
import pandas as pd
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import json
from datetime import datetime
import ast
from src.data.db import connect_to_db, process_list_column, insert_job_batch, insert_clean_job_batch
from tqdm import tqdm
from src.utils.helpers import clean_html, normalize_text, lemmatize_text, standardize_companyname, standardize_title

# Load environment variables
load_dotenv()

def extract_and_load_jobs_chunked(csv_file_path, chunk_size=10000):
    """Extract jobs from CSV file in chunks and load them into jobs table."""
    # Get database connection
    conn = connect_to_db()
    cursor = conn.cursor()
    
    print(f"Loading data from {csv_file_path} in chunks of {chunk_size}...")
    
    try:
        # Process the CSV in chunks to avoid loading the entire file
        total_processed = 0
        
        # Iterate through chunks of the CSV
        for chunk_df in pd.read_csv(csv_file_path, chunksize=chunk_size):
            chunk_size = len(chunk_df)
            
            # Process list-like columns vectorized
            list_columns = ['nlpBenefits', 'nlpSkills', 'nlpSoftSkills', 'nlpDegreeLevel']
            for col in list_columns:
                if col in chunk_df.columns:
                    chunk_df[col] = chunk_df[col].apply(process_list_column)
            
            # Replace NaT with None for correctDate
            if 'correctDate' in chunk_df.columns:
                chunk_df['correctDate'] = pd.to_datetime(chunk_df['correctDate'], errors='coerce')
                chunk_df['correctDate'] = chunk_df['correctDate'].where(chunk_df['correctDate'].notnull(), None)

            
            # Convert chunk to list of tuples for database insertion
            job_data = [
                (
                    row.get('lid'),
                    row.get('jobTitle'),
                    row.get('companyName'),
                    row.get('jobDescRaw'),
                    row.get('finalZipcode'),
                    row.get('finalState'),
                    row.get('finalCity'),
                    row.get('companyBranchName'),
                    row.get('jobDescUrl'),
                    row.get('nlpBenefits', []),
                    row.get('nlpSkills', []),
                    row.get('nlpSoftSkills', []),
                    row.get('nlpDegreeLevel', []),
                    row.get('nlpEmployment'),
                    row.get('nlpSeniority'),
                    row.get('correctDate') if pd.notnull(row.get('correctDate')) else None,
                    row.get('scrapedLocation')
                )
                for _, row in chunk_df.iterrows()
            ]
            
            # Insert chunk into database
            if insert_job_batch(cursor, job_data):
                total_processed += chunk_size
                print(f"Processed {total_processed} records so far...")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Successfully loaded all job data into the database")
        return True
        
    except Exception as e:
        print(f"Error processing CSV data: {e}")
        conn.close()
        return False


def extract_clean_jobs_data():
    """Fetch jobs data, preprocess it, and insert into jobs_processed table."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Fetch data from jobs table
    cursor.execute("""
    SELECT lid, jobtitle, companyname, finalcity, finalstate, finalzipcode, 
           jobdescraw, nlpskills, nlpsoftskills, nlpdegreelevel, nlpemployment, 
           nlpseniority, correctdate
    FROM jobs
    """)
    
    jobs = cursor.fetchall()
    print(f"Processing {len(jobs)} job records...")
    
    # Prepare data for batch insert
    processed_data = []
    
    for job in tqdm(jobs):
        lid, jobtitle, companyname, finalcity, finalstate, finalzipcode, jobdescraw, \
        nlpskills, nlpsoftskills, nlpdegreelevel, nlpemployment, nlpseniority, correctdate = job
        
        # Clean and normalize job description
        jobdesc_clean = clean_html(jobdescraw)
        
        # Lemmatize job description
        jobdesc_lemmatized = lemmatize_text(normalize_text(jobdesc_clean))
        
        # Standardize entities
        jobtitle_normalized = normalize_text(jobtitle)
        jobtitle_standardized = standardize_title(lemmatize_text(jobtitle_normalized))
        companyname_standardized = standardize_companyname(normalize_text(companyname))

        finalcity = normalize_text(finalcity)
        finalstate = normalize_text(finalstate)
        finalzipcode = normalize_text(finalzipcode)
        
        # Append processed record
        processed_data.append((
            lid, jobtitle, jobtitle_normalized, jobtitle_standardized, companyname, companyname_standardized,
            finalcity, finalstate, finalzipcode, jobdesc_clean, jobdesc_lemmatized,
            nlpskills, nlpsoftskills, nlpdegreelevel, nlpemployment, nlpseniority, correctdate
        ))
    
    if insert_clean_job_batch(cursor, processed_data): 
        conn.commit()
        print(f"Successfully processed and inserted {len(processed_data)} job records.")
    
    
    # Create indices for faster querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_jobtitle ON jobs_processed(jobtitle_normalized)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_companyname ON jobs_processed(companyname_standardized)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_location ON jobs_processed(finalcity, finalstate)")
    
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    setup_db_schema()
    extract_and_load_jobs_chunked(DATA_DIR / 'jobs.csv')
    extract_clean_jobs_data()
    
