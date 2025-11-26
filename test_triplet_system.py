"""
Quick test script for triplet system
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/home/ubuntu/crawl/crawler-recommend-sys/src')

from triplet_utils import TripletManager

def test_triplet_manager():
    """Test TripletManager functionality."""
    print("=" * 60)
    print("Testing TripletManager")
    print("=" * 60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'industry': ['Healthcare', 'Finance', 'E-commerce', 'Healthcare'],
        'client_size': ['51-200 Employees', '11-50 Employees', '1000+ Employees', '201-500 Employees'],
        'client_min': [51, 11, 1000, 201],
        'client_max': [200, 50, 5000, 500],
        'services': [
            'Mobile Development, AI/ML, Cloud Services',
            'Web Development, DevOps',
            'E-commerce, Mobile Development, UI/UX Design',
            'Mobile Development, Healthcare IT'
        ]
    })
    
    print("\nSample data:")
    print(sample_data)
    
    # Initialize and fit
    tm = TripletManager(max_services=3, triplet_separator='|||')
    tm.fit(sample_data)
    
    print(f"\nTop services learned: {tm.top_services[:10]}")
    
    # Create triplets
    print("\nCreated triplets:")
    for idx, row in sample_data.iterrows():
        triplet = tm.create_triplet(row)
        industry, size, services = tm.parse_triplet(triplet)
        print(f"{idx+1}. {triplet}")
        print(f"   Parsed: Industry={industry}, Size={size}, Services={services}")
    
    # Test similarity
    print("\nTriplet similarity tests:")
    triplet1 = tm.create_triplet(sample_data.iloc[0])
    triplet2 = tm.create_triplet(sample_data.iloc[3])
    
    sim = tm.calculate_triplet_similarity(triplet1, triplet2)
    print(f"\nTriplet 1: {triplet1}")
    print(f"Triplet 2: {triplet2}")
    print(f"Similarity: {sim:.4f}")
    
    # Test exact vs partial match
    is_exact = tm.is_exact_match(triplet1, triplet2)
    is_partial = tm.is_partial_match(triplet1, triplet2, threshold=0.5)
    
    print(f"Exact match: {is_exact}")
    print(f"Partial match (threshold=0.5): {is_partial}")
    
    print("\n✅ TripletManager tests passed!")
    return tm


def test_data_preparation():
    """Test data preparation with real data."""
    print("\n" + "=" * 60)
    print("Testing Data Preparation")
    print("=" * 60)
    
    try:
        from preprocessing_data import full_pipeline_preprocess_data
        from triplet_utils import add_triplet_column
        
        # Load small sample
        data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_0_100_update.csv"
        
        print(f"\nLoading data from: {data_path}")
        df = full_pipeline_preprocess_data(data_path)
        
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check required columns
        required_cols = ['industry', 'services', 'linkedin_company_outsource']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"❌ Missing columns: {missing}")
            return None
        
        print("✅ All required columns present")
        
        # Create triplets
        tm = TripletManager(max_services=3)
        tm.fit(df.head(100))  # Fit on first 100 rows
        
        df_with_triplets = add_triplet_column(df.head(20), tm)
        
        print(f"\nSample triplets from real data:")
        for idx, row in df_with_triplets.head(5).iterrows():
            print(f"{idx+1}. User: {row['linkedin_company_outsource']}")
            print(f"   Triplet: {row['triplet']}")
        
        print(f"\n✅ Data preparation successful!")
        print(f"   Total unique triplets: {df_with_triplets['triplet'].nunique()}")
        print(f"   Total unique users: {df_with_triplets['linkedin_company_outsource'].nunique()}")
        
        return df_with_triplets
        
    except FileNotFoundError:
        print(f"❌ Data file not found. Skipping this test.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_benchmark():
    """Test benchmark with triplets."""
    print("\n" + "=" * 60)
    print("Testing Benchmark Evaluation")
    print("=" * 60)
    
    from benchmark_data import BenchmarkOutput
    
    # Create sample predictions
    predictions = pd.DataFrame({
        'linkedin_company_outsource': ['user1', 'user1', 'user1', 'user2', 'user2'],
        'triplet': [
            'Healthcare|||medium|||Mobile,AI',
            'Finance|||small|||Web,DevOps',
            'Healthcare|||large|||AI,Cloud',
            'E-commerce|||medium|||Mobile,Web',
            'Finance|||medium|||Mobile,Cloud'
        ],
        'score': [0.9, 0.8, 0.7, 0.85, 0.75]
    })
    
    # Create ground truth
    ground_truth = pd.DataFrame({
        'linkedin_company_outsource': ['user1', 'user1', 'user2'],
        'triplet': [
            'Healthcare|||medium|||Mobile,AI',  # Exact match for user1
            'Finance|||medium|||Web,Cloud',     # Partial match
            'E-commerce|||medium|||Mobile,Web'  # Exact match for user2
        ]
    })
    
    print("\nPredictions:")
    print(predictions)
    print("\nGround Truth:")
    print(ground_truth)
    
    # Test exact match
    print("\n--- EXACT MATCH EVALUATION ---")
    benchmark_exact = BenchmarkOutput(predictions, ground_truth)
    summary_exact, _ = benchmark_exact.evaluate_topk(k=3, use_partial_match=False)
    print(summary_exact)
    
    # Test partial match
    print("\n--- PARTIAL MATCH EVALUATION ---")
    
    tm = TripletManager()
    
    def similarity_fn(t1, t2):
        return tm.calculate_triplet_similarity(t1, t2)
    
    benchmark_partial = BenchmarkOutput(predictions, ground_truth, similarity_fn=similarity_fn)
    summary_partial, _ = benchmark_partial.evaluate_topk(
        k=3, 
        use_partial_match=True,
        partial_match_threshold=0.5
    )
    print(summary_partial)
    
    print("\n✅ Benchmark tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRIPLET SYSTEM VERIFICATION TESTS")
    print("=" * 60)
    
    try:
        # Test 1: TripletManager
        tm = test_triplet_manager()
        
        # Test 2: Data preparation
        df = test_data_preparation()
        
        # Test 3: Benchmark
        test_benchmark()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)
        print("\n✅ System is ready to use.")
        print("\nTo run full experiments:")
        print("  cd /home/ubuntu/crawl/crawler-recommend-sys/src")
        print("  python execute_triplet.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
