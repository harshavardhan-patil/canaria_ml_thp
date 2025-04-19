import os
import pickle
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Tuple, Optional, Union
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer
from src.data.db_schema import setup_db_schema
from src.data.db import connect_to_db
from src.data.db import get_connection, release_connection


class VectorSearch:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', embedding_dim: int = 384):
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(self.embedding_model_name)

        # create embedding tables if not exist
        setup_db_schema()
    
    def generate_embeddings(self, text_list: list[str]) -> np.ndarray:
        """
        Generate an embedding for the given text.
        """        
        return self.model.encode(text_list)

    def add_batch(self, job_embeddings: list[str, str]) -> bool:
        """
        Add a batch of job embeddings to the database.
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            execute_values(
            cursor,
            """INSERT INTO job_embeddings 
            (lid, embedding) VALUES %s 
            ON CONFLICT (lid) DO UPDATE SET embedding = EXCLUDED.embedding""",
            [(lid, embedding.tolist()) for lid, embedding in job_embeddings],
            template="(%s, %s::vector)"
            )

            cursor.execute("SELECT COUNT(*) FROM job_embeddings")
            conn.commit()
            print(f"Added {len(job_embeddings)} job embeddings to database.")

            return True
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error adding batch of embeddings: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)

    def remove_batch(self, lids: List[str]) -> bool:
        """
        Remove a batch of job embeddings from the database based on the lid.
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Delete embeddings in batch
            placeholders = ','.join(['%s'] * len(lids))
            cursor.execute(f"DELETE FROM job_embeddings WHERE lid IN ({placeholders})", lids)
            
            # Also delete any duplicate pairs involving these jobs
            cursor.execute(f"""
                DELETE FROM job_duplicates 
                WHERE lid1 IN ({placeholders}) OR lid2 IN ({placeholders})
            """, lids + lids)
            
            count = cursor.rowcount
            conn.commit()
            print(f"Removed {count} job embeddings from database.")
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error removing batch of embeddings: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)

    def create_index(self,):
        """
        Create IVFFLAT index. Needs to be ran after some data in loaded in the table.
        Alternatively could use HNSW
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS job_embeddings_idx ON job_embeddings 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)

            conn.commit()
            print(f"Generated IVFFLAT index")
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error adding batch of embeddings: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)

    def search(self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.8) -> List[Dict]:
        """
        Search for similar job embeddings.
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Search for similar embeddings using L2 distance
            cursor.execute("""
                SELECT 
                    lid, 
                    1 - (embedding <=> %s::vector) as similarity
                FROM job_embeddings
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), threshold, k))
            
            results = [{'lid': lid, 'similarity': float(similarity)} for lid, similarity in cursor.fetchall()]
            print(f"Found {len(results)} similar job embeddings.")
            return results
            
        except Exception as e:
            print(f"Error searching for similar embeddings: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)

    def store_duplicates(self, duplicates: List[Tuple[str, str, float]]):
        """
        Store duplicate job pairs in the job_duplicates table.
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Insert duplicates in batch
            execute_values(
                cursor,
                """
                INSERT INTO job_duplicates (lid1, lid2, similarity_score)
                VALUES %s
                ON CONFLICT (lid1, lid2) DO UPDATE SET 
                    similarity_score = EXCLUDED.similarity_score,
                    created_at = CURRENT_TIMESTAMP
                """,
                [(lid1, lid2, score) for lid1, lid2, score in duplicates],
                template="(%s, %s, %s)"
            )
            
            count = len(duplicates)
            conn.commit()
            #print(f"Stored {count} duplicate job pairs in database.")
            return count
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error storing duplicate jobs: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)

    def find_duplicates_by_exact_match(self, lids: list[str]):
        """
        Find duplicate jobs by exact match of title, company and location
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            lids = list(lids)
            placeholders = ', '.join(['%s'] * len(lids))

            query = f"""
                SELECT j1.lid, j2.lid, 1.0
                FROM jobs j1
                JOIN jobs j2 ON 
                    (j1.jobTitle = j2.jobTitle AND
                    j1.companyName = j2.companyName AND
                    j1.finalCity = j2.finalCity AND
                    j1.finalState = j2.finalState AND
                    j1.finalzipcode = j2.finalzipcode AND
                    j1.lid <> j2.lid)
                WHERE j1.lid IN ({placeholders})
            """

            cursor.execute(query, lids)
            duplicates = cursor.fetchall()

            #print(f"Found {len(duplicates)} duplicate job pairs by exact match.")
            return duplicates
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error finding duplicate jobs by exact match: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)
    
    def filter_jobs_by_location(self, lids: list[str]) -> List[str]:
        """
        Find jobs by location to further process for duplicates.
        Note: Switching to zipcode only can heavily reduce the number of potentials...depends on how reliable zipcode field is
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            placeholders = ','.join(["%s"] * len(lids))
            query = f"""
                SELECT j2.lid
                FROM jobs_processed j2
                JOIN jobs_processed j1
                ON (j1.finalcity = j2.finalcity AND
                    j1.finalstate = j2.finalstate OR
                    j1.finalzipcode = j2.finalzipcode)
                    OR
                    (j2.finalcity IS NULL OR
                        j2.finalstate IS NULL OR
                        j2.finalzipcode IS NULL)
                WHERE j1.lid IN ({placeholders})
            """

            cursor.execute(query, lids)
            job_ids = [row[0] for row in cursor.fetchall()]
            #print(f"Found {len(job_ids)} jobs after location filtering.")
            return job_ids
            
        except Exception as e:
            print(f"Error filtering jobs by location: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)
    
    def filter_jobs_by_location_and_company(self, lids: list[str]) -> List[str]:
        """
        Find jobs by location to further process for duplicates. 
        Note: Using standardized companyname's could have edge case misses
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            placeholders = ','.join(["%s"] * len(lids))
            query = f"""
                SELECT j2.lid
                FROM jobs_processed j2
                JOIN jobs_processed j1
                ON ((j1.finalcity = j2.finalcity AND
                    j1.finalstate = j2.finalstate OR
                    j1.finalzipcode = j2.finalzipcode)
                    OR
                    (j2.finalcity IS NULL OR
                        j2.finalstate IS NULL OR
                        j2.finalzipcode IS NULL))
                    AND
                    j1.companyname_standardized = j2.companyname_standardized
                WHERE j1.lid IN ({placeholders})
            """

            cursor.execute(query, lids)
            job_ids = [row[0] for row in cursor.fetchall()]
            #print(f"Found {len(job_ids)} jobs after location filtering.")
            return job_ids
            
        except Exception as e:
            print(f"Error filtering jobs by location: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)
    
    def filter_jobs_by_nlp_attributes(self, lids: list[str]) -> List[str]:
        """
        Find jobs with similar NLP attributes to further process for duplicates.
        This filters based on skills, soft skills, degree level, employment type and seniority.
        At least one match filtering, need to experiment with strict filtering
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            placeholders = ','.join(["%s"] * len(lids))
            query = f"""
                SELECT j2.lid
                FROM jobs_processed j2
                JOIN jobs_processed j1
                ON (
                    j1.nlpskills && j2.nlpskills
                    AND
                    j1.nlpsoftskills && j2.nlpsoftskills
                    AND
                    j1.nlpdegreelevel && j2.nlpdegreelevel
                    AND
                    j1.nlpemployment = j2.nlpemployment
                    AND
                    j1.nlpseniority = j2.nlpseniority
                )
                WHERE j1.lid IN ({placeholders})
            """

            cursor.execute(query, lids)
            job_ids = [row[0] for row in cursor.fetchall()]
            #print(f"Found {len(job_ids)} jobs after NLP attributes filtering.")
            return job_ids
            
        except Exception as e:
            print(f"Error filtering jobs by NLP attributes: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)

    def find_duplicates_for_batch(self, lids: List[str], threshold: float = 0.824) -> List[Tuple[str, str, float]]:
        """
         Find duplicate jobs by embedding similarity among a list of job IDs.
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Create a temporary table with the job IDs to process
            cursor.execute("CREATE TEMPORARY TABLE temp_job_ids (lid VARCHAR(255) PRIMARY KEY)")
            execute_values(
                cursor, 
                "INSERT INTO temp_job_ids (lid) VALUES %s ON CONFLICT DO NOTHING",
                [(job_id,) for job_id in lids],
                template="(%s)"
            )

            cursor.execute("""
                SELECT e1.lid, e2.lid, 1 - (e1.embedding <-> e2.embedding) as similarity
                FROM job_embeddings e1
                JOIN temp_job_ids t1 ON e1.lid = t1.lid
                JOIN job_embeddings e2 ON e2.lid IN (SELECT lid FROM temp_job_ids)
                WHERE e1.lid <> e2.lid
                AND 1 - (e1.embedding <=> e2.embedding) >= %s
            """, (threshold,))

            duplicates = cursor.fetchall()
            #print(f"Found {len(duplicates)} duplicate job pairs by similarity search.")
            return duplicates

        
        except Exception as e:
            print(f"Error filtering jobs by location: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)
    
    def find_duplicates_for_job(self, lid: str, potentials: List[str], threshold: float = 0.824) -> List[Tuple[str, str, float]]:
        """
         Find duplicate jobs by embedding similarity among a list of potentials for a specific lid
        """
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            

            query = """
            WITH original_embedding AS (
                SELECT embedding FROM job_embeddings WHERE lid = %s
            )
            SELECT 
                %s AS original_lid,
                e.lid AS duplicate_lid, 
                1 - (e.embedding <=> (SELECT embedding FROM original_embedding)) AS similarity
            FROM job_embeddings e
            WHERE e.lid = ANY(%s)
            AND e.lid <> %s
            AND 1 - (e.embedding <=> (SELECT embedding FROM original_embedding)) >= %s
            ORDER BY similarity DESC
            """
            
            cursor.execute(query, (lid, lid, potentials, lid, threshold))
            duplicates = cursor.fetchall()
            
            #print(f"Found {len(duplicates)} duplicate jobs for {lid} with similarity >= {threshold}")
            return duplicates

        
        except Exception as e:
            print(f"Error filtering jobs by location: {e}")
            raise
        finally:
            if conn:
                release_connection(conn)