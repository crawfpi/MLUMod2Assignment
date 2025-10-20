# MLU Module 2 Assignment: Hierarchical Clustering vs K-Means

## Overview
This assignment compares the performance and behavior of Hierarchical Clustering and K-Means clustering algorithms on various synthetic datasets.

## Environment Specifications
- **Python Version**: 3.13.2 (Development: 3.12.x)
- **scikit-learn Version**: 1.7.2
- **Reference Lab Version**: Python 3.7.6, scikit-learn 0.22.2post1

## Installation

### Prerequisites
Ensure you have Python 3.12+ installed on your system.

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/crawfpi/MLUMod2Assignment.git
   cd MLUMod2Assignment
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the clustering comparison script:
```bash
python clustering_comparison.py
```

This will:
1. Generate multiple synthetic datasets (blobs, moons, circles, anisotropic)
2. Apply both K-Means and Hierarchical clustering
3. Evaluate performance using multiple metrics (Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index)
4. Generate visualization plots saved as `clustering_comparison_results.png`
5. Display a comprehensive comparison table

## Output

The script generates:
- **Console Output**: Detailed comparison table with metrics for each algorithm
- **Visualization**: `clustering_comparison_results.png` showing clustering results

## Metrics Explained

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1], higher is better.
- **Davies-Bouldin Index**: Average similarity measure of each cluster with its most similar cluster. Lower values indicate better clustering.
- **Adjusted Rand Index**: Measures similarity between predicted and true labels. Range: [-1, 1], higher is better.

## Key Findings

### K-Means
- ✅ Faster execution time
- ✅ Works well with spherical, well-separated clusters
- ⚠️ Sensitive to initialization
- ⚠️ Assumes equal cluster sizes

### Hierarchical Clustering
- ✅ No need to specify cluster count initially
- ✅ Better for complex cluster shapes
- ⚠️ Computationally expensive for large datasets
- ✅ Deterministic (no random initialization)

## License
Educational use - MLU Module 2 Assignment
