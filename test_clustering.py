"""
Unit tests for clustering comparison functionality.

Tests the core functions of the clustering_comparison module.
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs
from clustering_comparison import (
    generate_datasets,
    perform_clustering,
    evaluate_clustering
)


class TestClusteringComparison(unittest.TestCase):
    """Test cases for clustering comparison functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X, self.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    def test_generate_datasets(self):
        """Test that datasets are generated correctly."""
        datasets = generate_datasets()
        
        # Check that all expected datasets are present
        expected_datasets = {'blobs', 'moons', 'circles', 'anisotropic'}
        self.assertEqual(set(datasets.keys()), expected_datasets)
        
        # Check that each dataset has the correct structure
        for name, (X, y) in datasets.items():
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual(X.shape[0], y.shape[0])  # Same number of samples
            self.assertEqual(X.shape[1], 2)  # 2D features
    
    def test_kmeans_clustering(self):
        """Test K-Means clustering."""
        labels, time_taken = perform_clustering(self.X, n_clusters=3, method='kmeans')
        
        # Check output types
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(time_taken, float)
        
        # Check labels properties
        self.assertEqual(len(labels), len(self.X))
        self.assertTrue(len(np.unique(labels)) <= 3)  # At most 3 clusters
        self.assertTrue(time_taken >= 0)  # Non-negative time
    
    def test_hierarchical_clustering(self):
        """Test Hierarchical clustering."""
        labels, time_taken = perform_clustering(self.X, n_clusters=3, method='hierarchical')
        
        # Check output types
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(time_taken, float)
        
        # Check labels properties
        self.assertEqual(len(labels), len(self.X))
        self.assertTrue(len(np.unique(labels)) <= 3)  # At most 3 clusters
        self.assertTrue(time_taken >= 0)  # Non-negative time
    
    def test_clustering_invalid_method(self):
        """Test that invalid clustering method raises error."""
        with self.assertRaises(ValueError):
            perform_clustering(self.X, n_clusters=3, method='invalid_method')
    
    def test_evaluate_clustering(self):
        """Test clustering evaluation metrics."""
        labels, _ = perform_clustering(self.X, n_clusters=3, method='kmeans')
        metrics = evaluate_clustering(self.X, labels, self.y)
        
        # Check that all expected metrics are present
        self.assertIn('silhouette', metrics)
        self.assertIn('davies_bouldin', metrics)
        self.assertIn('adjusted_rand', metrics)
        
        # Check metric ranges
        self.assertTrue(-1 <= metrics['silhouette'] <= 1)
        self.assertTrue(metrics['davies_bouldin'] >= 0)
        self.assertTrue(-1 <= metrics['adjusted_rand'] <= 1)
    
    def test_evaluate_clustering_without_ground_truth(self):
        """Test evaluation without ground truth labels."""
        labels, _ = perform_clustering(self.X, n_clusters=3, method='kmeans')
        metrics = evaluate_clustering(self.X, labels)
        
        # Should have silhouette and davies_bouldin, but not adjusted_rand
        self.assertIn('silhouette', metrics)
        self.assertIn('davies_bouldin', metrics)
        self.assertNotIn('adjusted_rand', metrics)
    
    def test_clustering_reproducibility(self):
        """Test that K-Means with same random_state gives same results."""
        labels1, _ = perform_clustering(self.X, n_clusters=3, method='kmeans')
        labels2, _ = perform_clustering(self.X, n_clusters=3, method='kmeans')
        
        # Results should be identical with same random state
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_different_cluster_numbers(self):
        """Test clustering with different numbers of clusters."""
        for n_clusters in [2, 3, 4, 5]:
            labels_km, _ = perform_clustering(self.X, n_clusters=n_clusters, method='kmeans')
            labels_hc, _ = perform_clustering(self.X, n_clusters=n_clusters, method='hierarchical')
            
            self.assertTrue(len(np.unique(labels_km)) <= n_clusters)
            self.assertTrue(len(np.unique(labels_hc)) <= n_clusters)


class TestDatasetProperties(unittest.TestCase):
    """Test properties of generated datasets."""
    
    def test_dataset_sizes(self):
        """Test that all datasets have the expected size."""
        datasets = generate_datasets()
        
        for name, (X, y) in datasets.items():
            self.assertEqual(X.shape[0], 300, f"Dataset {name} should have 300 samples")
    
    def test_dataset_dimensions(self):
        """Test that all datasets are 2-dimensional."""
        datasets = generate_datasets()
        
        for name, (X, y) in datasets.items():
            self.assertEqual(X.shape[1], 2, f"Dataset {name} should have 2 features")


if __name__ == '__main__':
    unittest.main()
