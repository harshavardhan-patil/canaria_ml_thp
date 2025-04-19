from src.vector_search import VectorSearch

vector_search = VectorSearch()

def check_duplicates_for_lid(lid: str):
    """
    Returns duplicates with similarity scores
    """
    try:
        job_id = lid

        # Find potential jobs to compare based on location filtering
        filtered_job_ids = vector_search.filter_jobs_by_location_and_company([job_id])

        # Find duplicates by similarity
        similarity_duplicates = vector_search.find_duplicates_for_job_hierarchial(
            job_id,
            filtered_job_ids,
        )
    
        return similarity_duplicates
    
    except Exception as e:
        print(f"Error during processing: {e}")