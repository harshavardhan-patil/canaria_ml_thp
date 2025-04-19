import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
from src.data.db import connect_to_db

#############################################################
# 10000 jobs run output #
#
# Similarity Distribution Statistics:
# Number of pairs analyzed: 49995000
# Min: -0.205
# Max: 1.000
# Mean: 0.286
# Median: 0.273
# Standard Deviation: 0.123
#
# The distribution appears to be bimodal with 5 peaks.
# Suggested threshold based on valley between peaks: 0.824
#
# Final recommended threshold: 0.824
################################################################

def analyze_similarity_distribution(sample_size=10000):
    """
    Analyze the distribution of pairwise similarities between job embeddings
    to help determine an appropriate threshold for duplicate detection.
    """
    # Connect to database
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Sample random job embeddings
    print(f"Sampling {sample_size} job embeddings...")
    try:
        cursor.execute(f"""
            SELECT lid, embedding
            FROM job_embeddings
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """)
    except Exception as e:
        print(f"Error with RANDOM(): {e}")
        try:
            cursor.execute(f"""
                SELECT lid, embedding
                FROM job_embeddings TABLESAMPLE SYSTEM(
                    {min(100, int(100 * sample_size / 10000))}
                )
                LIMIT {sample_size}
            """)
        except Exception as e2:
            print(f"Error with TABLESAMPLE: {e2}")
            cursor.execute(f"""
                SELECT lid, embedding
                FROM job_embeddings
                LIMIT {sample_size}
            """)
    
    jobs = cursor.fetchall()
    job_ids = [job[0] for job in jobs]
    
    # Convert embeddings to numeric arrays
    embeddings = []
    for job in jobs:
        if isinstance(job[1], str):
            import json
            try:
                embedding = np.array(json.loads(job[1]))
            except json.JSONDecodeError:
                embedding = np.fromstring(job[1].strip('[]'), sep=',')
        else:
            embedding = np.array(job[1])
        
        embeddings.append(embedding)
    
    print(f"Calculating pairwise similarities for {len(jobs)} jobs...")
    
    similarities = []
    
    for i in tqdm(range(len(embeddings))):
        # Calculate similarity with remaining jobs (avoid duplicate comparisons)
        for j in range(i+1, len(embeddings)):
            # Skip if vectors are incompatible (different sizes)
            if embeddings[i].shape != embeddings[j].shape:
                continue
                
            # Cosine similarity: dot product of normalized vectors
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            
            # Check for zero norms to avoid division by zero
            if norm_i == 0 or norm_j == 0:
                sim = 0
            else:
                sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)

            similarities.append(sim)
                
    
    similarities = np.array(similarities)
    
    # Generate statistics
    mean = np.mean(similarities)
    median = np.median(similarities)
    std = np.std(similarities)
    min_val = np.min(similarities)
    max_val = np.max(similarities)
    
    # Check if we have enough similarity data to proceed
    if len(similarities) < 10:
        print("Not enough similarity data to analyze distribution.")
        return {
            'min': min_val if 'min_val' in locals() else 0,
            'max': max_val if 'max_val' in locals() else 0,
            'mean': mean if 'mean' in locals() else 0,
            'median': median if 'median' in locals() else 0,
            'std': std if 'std' in locals() else 0,
            'otsu_threshold': 0.8,  # Default threshold
            'suggested_threshold': 0.8  # Default threshold
        }
    
    # Check if distribution appears to be bimodal
    kde = stats.gaussian_kde(similarities)
    x = np.linspace(min_val, max_val, 1000)
    y = kde(x)
    peaks, _ = find_peaks(y)
    
    # Find potential threshold using Otsu's method (commonly used for image thresholding)
    hist, bin_edges = np.histogram(similarities, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    otsu_threshold = threshold_otsu(hist, bin_centers)
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Histogram with KDE
    plt.subplot(2, 1, 1)
    sns.histplot(similarities, bins=100, kde=True)
    plt.axvline(x=median, color='r', linestyle='--', label=f'Median: {median:.3f}')
    plt.axvline(x=mean, color='g', linestyle='--', label=f'Mean: {mean:.3f}')
    plt.axvline(x=otsu_threshold, color='purple', linestyle='--', 
                label=f'Otsu Threshold: {otsu_threshold:.3f}')
    
    # Mark peaks if distribution appears bimodal
    if len(peaks) > 1:
        for peak in peaks:
            plt.axvline(x=x[peak], color='orange', alpha=0.5)
        
        # Suggest threshold as the minimum between the two highest peaks
        peak_heights = y[peaks]
        top_peaks_idx = np.argsort(peak_heights)[-2:]
        top_peaks = peaks[top_peaks_idx]
        top_peaks.sort()
        
        # Find minimum between the two peaks
        between_peaks = y[top_peaks[0]:top_peaks[1]]
        min_idx = np.argmin(between_peaks) + top_peaks[0]
        suggested_threshold = x[min_idx]
        
        plt.axvline(x=suggested_threshold, color='blue', linestyle='-.',
                   label=f'Suggested Threshold: {suggested_threshold:.3f}')
    
    plt.title('Distribution of Pairwise Similarities Between Job Embeddings')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Zoomed version focusing on high similarity region
    plt.subplot(2, 1, 2)
    high_sim_threshold = max(0.5, median)  # Focus on higher similarities
    high_similarities = similarities[similarities >= high_sim_threshold]
    
    if len(high_similarities) > 0:
        sns.histplot(high_similarities, bins=50, kde=True)
        plt.axvline(x=median, color='r', linestyle='--', label=f'Median: {median:.3f}')
        plt.axvline(x=mean, color='g', linestyle='--', label=f'Mean: {mean:.3f}')
        plt.axvline(x=otsu_threshold, color='purple', linestyle='--', 
                    label=f'Otsu Threshold: {otsu_threshold:.3f}')
        
        if len(peaks) > 1 and 'suggested_threshold' in locals():
            plt.axvline(x=suggested_threshold, color='blue', linestyle='-.',
                       label=f'Suggested Threshold: {suggested_threshold:.3f}')
        
        plt.title('Distribution of High Similarity Scores (Zoomed)')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('similarity_distribution.png')
    plt.close()
    
    # Print statistics
    print("\nSimilarity Distribution Statistics:")
    print(f"Number of pairs analyzed: {len(similarities)}")
    print(f"Min: {min_val:.3f}")
    print(f"Max: {max_val:.3f}")
    print(f"Mean: {mean:.3f}")
    print(f"Median: {median:.3f}")
    print(f"Standard Deviation: {std:.3f}")
    
    if len(peaks) > 1 and 'suggested_threshold' in locals():
        print(f"\nThe distribution appears to be bimodal with {len(peaks)} peaks.")
        print(f"Suggested threshold based on valley between peaks: {suggested_threshold:.3f}")
    else:
        print("\nThe distribution does not appear to be clearly bimodal.")
        print(f"Suggested threshold (Otsu): {otsu_threshold:.3f}")

    # Close database connection
    conn.close()
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean,
        'median': median,
        'std': std,
        'otsu_threshold': otsu_threshold,
        'suggested_threshold': suggested_threshold if len(peaks) > 1 and 'suggested_threshold' in locals() else otsu_threshold
    }

def threshold_otsu(hist, bin_centers):
    """
    Implementation of Otsu's method for finding optimal threshold.
    Adapted from scikit-image implementation.
    """
    total = sum(hist)
    
    # Initialize variables
    weight1 = 0
    sum1 = 0
    max_variance = 0
    threshold = 0
    
    for i in range(len(hist)):
        weight1 += hist[i]
        if weight1 == 0:
            continue
            
        weight2 = total - weight1
        if weight2 == 0:
            break
            
        sum1 += hist[i] * bin_centers[i]
        mean1 = sum1 / weight1
        mean2 = (sum(hist * bin_centers) - sum1) / weight2
        
        # Calculate between-class variance
        variance = weight1 * weight2 * (mean1 - mean2) ** 2
        
        # Update threshold if variance is greater
        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]
    
    return threshold

if __name__ == "__main__":
    results = analyze_similarity_distribution()
    
    # Print final recommended threshold
    print(f"\nFinal recommended threshold: {results['suggested_threshold']:.3f}")