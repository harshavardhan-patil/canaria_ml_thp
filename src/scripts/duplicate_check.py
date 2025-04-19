from src.data.db import get_jobs
from src.vector_search import VectorSearch
from src.utils.helpers import get_batches
from tqdm import tqdm
import argparse

def generate_embeddings_for_all_data():
    # Initialize VectorSearch
    vector_search = VectorSearch()

    batch_size = 5000
    try:
        jobs = get_jobs()
        for jobs_batch in get_batches(jobs, batch_size):
            job_ids = [job[0] for job in jobs_batch]
            jd = [" ".join([job[1], job[2], job[3], job[4], job[5], job[6]]) for job in jobs_batch]
            job_titles = [job[1] for job in jobs_batch]
            
            # Generate embeddings for job descriptions
            embeddings = vector_search.generate_embeddings(jd)
            
            # Add jobs embeddings to database
            job_embeddings = [(job_ids[i], embeddings[i]) for i in range(len(job_ids))]
            vector_search.add_batch(job_embeddings)

            # Generate embeddings for job titles
            title_embeddings = vector_search.generate_embeddings(job_titles)

            # Add title embeddings to database
            title_embeddings_batch = [(job_ids[i], title_embeddings[i]) for i in range(len(job_ids))]
            vector_search.add_title_batch(title_embeddings_batch)

        vector_search.create_index()
        vector_search.create_title_index()
        print(f"Embedded {len(jobs)} jobs")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise

def check_duplicates_for_all_data():
    jobs = get_jobs()
    vector_search = VectorSearch()
    total_duplicates_found = 0
    duplicates = []
    for job in tqdm(jobs):
        try:
            job_id = job[0]

            # Find potential jobs to compare based on location filtering
            filtered_job_ids = vector_search.filter_jobs_by_location_and_company([job_id])

            # Further filter based on similar nlp matching - array comparison query is super slow
            # filtered_job_ids = vector_search.filter_jobs_by_nlp_attributes(job_id, filtered_job_ids)

            # Find duplicates by similarity
            similarity_duplicates = vector_search.find_duplicates_for_job_hierarchial(
                job_id,
                filtered_job_ids,
            )
            # Store duplicates
            if similarity_duplicates:
                duplicates.extend(similarity_duplicates)
                total_duplicates_found += len(similarity_duplicates)
            
        except Exception as e:
            print(f"Error during processing: {e.with_traceback()}")
            break

    vector_search.store_duplicates(duplicates)
    print(f"Total duplicates found: {total_duplicates_found}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check for duplicate job postings')
    parser.add_argument('--generate-embeddings', action='store_true', 
                        help='Generate embeddings for all data (default: False)')
    args = parser.parse_args()
    
    if args.generate_embeddings:
        generate_embeddings_for_all_data()
    
    check_duplicates_for_all_data()