"""
MLU Mod2 Assignment: Hierarchical Clustering vs K-Means Comparison

This script compares the performance and behavior of Hierarchical Clustering
and K-Means clustering algorithms on synthetic datasets.

Author: MLU Module 2 Assignment
Python Version: 3.13.2
scikit-learn Version: 1.7.2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import time
import warnings

warnings.filterwarnings('ignore')


def generate_datasets():
    """
    Generate different types of synthetic datasets for clustering comparison.
    
    Returns:
        dict: Dictionary containing different datasets with their names
    """
    np.random.seed(42)
    
    datasets = {}
    
    # Dataset 1: Well-separated blobs
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                   cluster_std=0.5, random_state=42)
    datasets['blobs'] = (X_blobs, y_blobs)
    
    # Dataset 2: Moons (non-linear separation)
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
    datasets['moons'] = (X_moons, y_moons)
    
    # Dataset 3: Circles (concentric)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
                                         factor=0.5, random_state=42)
    datasets['circles'] = (X_circles, y_circles)
    
    # Dataset 4: Anisotropic blobs
    X_aniso, y_aniso = make_blobs(n_samples=300, centers=3, n_features=2, 
                                   random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    datasets['anisotropic'] = (X_aniso, y_aniso)
    
    return datasets


def perform_clustering(X, n_clusters, method='kmeans'):
    """
    Perform clustering using the specified method.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        method: 'kmeans' or 'hierarchical'
    
    Returns:
        tuple: (labels, execution_time)
    """
    start_time = time.time()
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    labels = clusterer.fit_predict(X)
    execution_time = time.time() - start_time
    
    return labels, execution_time


def evaluate_clustering(X, labels, true_labels=None):
    """
    Evaluate clustering results using multiple metrics.
    
    Args:
        X: Feature matrix
        labels: Predicted cluster labels
        true_labels: Ground truth labels (optional)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # Silhouette Score (higher is better, range [-1, 1])
    if len(set(labels)) > 1:
        metrics['silhouette'] = silhouette_score(X, labels)
    else:
        metrics['silhouette'] = -1
    
    # Davies-Bouldin Index (lower is better, minimum 0)
    if len(set(labels)) > 1:
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    else:
        metrics['davies_bouldin'] = float('inf')
    
    # Adjusted Rand Index (if ground truth available)
    if true_labels is not None:
        metrics['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
    
    return metrics


def plot_clustering_results(datasets, results):
    """
    Plot clustering results for visualization.
    
    Args:
        datasets: Dictionary of datasets
        results: Dictionary of clustering results
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 5 * n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, (X, y_true)) in enumerate(datasets.items()):
        # Original data
        axes[idx, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='k')
        axes[idx, 0].set_title(f'{name.title()}\n(Ground Truth)')
        axes[idx, 0].set_xlabel('Feature 1')
        axes[idx, 0].set_ylabel('Feature 2')
        
        # K-Means results
        kmeans_labels = results[name]['kmeans']['labels']
        axes[idx, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='k')
        axes[idx, 1].set_title(f'{name.title()}\n(K-Means)')
        axes[idx, 1].set_xlabel('Feature 1')
        axes[idx, 1].set_ylabel('Feature 2')
        
        # Hierarchical results
        hier_labels = results[name]['hierarchical']['labels']
        axes[idx, 2].scatter(X[:, 0], X[:, 1], c=hier_labels, cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='k')
        axes[idx, 2].set_title(f'{name.title()}\n(Hierarchical)')
        axes[idx, 2].set_xlabel('Feature 1')
        axes[idx, 2].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'clustering_comparison_results.png'")
    plt.close()


def print_comparison_table(results):
    """
    Print a formatted comparison table of clustering results.
    
    Args:
        results: Dictionary of clustering results
    """
    print("\n" + "="*100)
    print("CLUSTERING COMPARISON RESULTS")
    print("="*100)
    
    for dataset_name, methods in results.items():
        print(f"\n{dataset_name.upper()} DATASET")
        print("-"*100)
        print(f"{'Method':<20} {'Time (s)':<12} {'Silhouette':<15} {'Davies-Bouldin':<18} {'Adj. Rand Index':<15}")
        print("-"*100)
        
        for method_name, data in methods.items():
            metrics = data['metrics']
            time_taken = data['time']
            silhouette = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] != -1 else "N/A"
            davies = f"{metrics['davies_bouldin']:.4f}" if metrics['davies_bouldin'] != float('inf') else "N/A"
            adj_rand = f"{metrics.get('adjusted_rand', 0):.4f}" if 'adjusted_rand' in metrics else "N/A"
            
            print(f"{method_name.title():<20} {time_taken:<12.6f} {silhouette:<15} {davies:<18} {adj_rand:<15}")
        
        print()


def main():
    """
    Main function to run the clustering comparison.
    """
    print("="*100)
    print("MLU Module 2 Assignment: Hierarchical Clustering vs K-Means")
    print("="*100)
    
    # Check versions
    import sklearn
    import sys
    print(f"\nPython Version: {sys.version}")
    print(f"scikit-learn Version: {sklearn.__version__}")
    print()
    
    # Generate datasets
    print("Generating synthetic datasets...")
    datasets = generate_datasets()
    print(f"Generated {len(datasets)} datasets: {', '.join(datasets.keys())}")
    
    # Determine number of clusters for each dataset
    n_clusters_map = {
        'blobs': 4,
        'moons': 2,
        'circles': 2,
        'anisotropic': 3
    }
    
    # Perform clustering and evaluation
    print("\nPerforming clustering analysis...\n")
    results = {}
    
    for name, (X, y_true) in datasets.items():
        print(f"Processing {name} dataset...")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = n_clusters_map[name]
        results[name] = {}
        
        # K-Means
        kmeans_labels, kmeans_time = perform_clustering(X_scaled, n_clusters, 'kmeans')
        kmeans_metrics = evaluate_clustering(X_scaled, kmeans_labels, y_true)
        results[name]['kmeans'] = {
            'labels': kmeans_labels,
            'time': kmeans_time,
            'metrics': kmeans_metrics
        }
        
        # Hierarchical
        hier_labels, hier_time = perform_clustering(X_scaled, n_clusters, 'hierarchical')
        hier_metrics = evaluate_clustering(X_scaled, hier_labels, y_true)
        results[name]['hierarchical'] = {
            'labels': hier_labels,
            'time': hier_time,
            'metrics': hier_metrics
        }
    
    # Print comparison table
    print_comparison_table(results)
    
    # Plot results
    print("\nGenerating visualizations...")
    datasets_for_plot = {name: (X, y) for name, (X, y) in datasets.items()}
    plot_clustering_results(datasets_for_plot, results)
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print("\nKey Observations:")
    print("1. K-Means:")
    print("   - Generally faster for large datasets")
    print("   - Works well with spherical, well-separated clusters")
    print("   - Sensitive to initialization (uses k-means++ by default)")
    print("   - Assumes clusters are roughly equal in size and density")
    
    print("\n2. Hierarchical Clustering:")
    print("   - Does not require specifying number of clusters beforehand")
    print("   - Can capture more complex cluster shapes")
    print("   - More computationally expensive (O(n^2) vs O(n) for K-Means)")
    print("   - Does not require random initialization")
    
    print("\n3. When to use each:")
    print("   - Use K-Means: Large datasets, spherical clusters, speed is important")
    print("   - Use Hierarchical: Small to medium datasets, complex shapes, dendrograms needed")
    print("="*100)


if __name__ == "__main__":
    main()
