from src.vector_search import VectorSearch

def check_duplicates_for_lid(lid: str, threshold=0.824):
    vector_search = VectorSearch()
    try:
        job_id = lid

        # Find potential jobs to compare based on location filtering
        filtered_job_ids = vector_search.filter_jobs_by_location([job_id])

        # Find duplicates by similarity
        similarity_duplicates = vector_search.find_duplicates_for_job(
            job_id,
            filtered_job_ids, 
            threshold=threshold
        )
    
        return similarity_duplicates
    
    except Exception as e:
        print(f"Error during processing: {e}")