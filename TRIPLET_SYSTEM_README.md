# Triplet-Based Recommendation System

## 🎯 Overview

This system extends the original recommendation approach from single-item (industry) to **triplet-based** recommendations: `(Industry, Client Size, Services)`.

### Why Triplets?

Instead of recommending just:
- ❌ "Healthcare"

We now recommend specific, actionable leads:
- ✅ "Healthcare | Medium (51-200) | Mobile Development, AI/ML"

This provides:
- **Specificity**: Know exactly what type of clients to target
- **Actionability**: Clear understanding of client segment
- **Better Matching**: Avoid mismatches (e.g., enterprise leads for startups)

---

## 🏗️ Architecture

### 1. Triplet Creation (`triplet_utils.py`)

**TripletManager** handles:
- **Client Size Normalization**: `51-200 Employees` → `medium`
  - Buckets: `micro` (0-10), `small` (11-50), `medium` (51-200), `large` (201-1000), `enterprise` (1000+)
  
- **Service Normalization**: Extract top-3 most relevant services
  - Learns from training data (avoid data leakage)
  - Canonical mapping: "Mobile App Dev" → "Mobile Development"
  
- **Triplet Format**: `industry|||client_size|||services`
  - Example: `Healthcare|||medium|||Mobile Development,AI/ML,Cloud Services`

**Key Features**:
```python
# Fit on training data only
triplet_manager = TripletManager(max_services=3)
triplet_manager.fit(df_train)

# Create triplets
df['triplet'] = df.apply(triplet_manager.create_triplet, axis=1)

# Calculate similarity
similarity = triplet_manager.calculate_triplet_similarity(
    triplet1, triplet2,
    weights=(0.5, 0.3, 0.2)  # industry, size, services
)
```

---

### 2. User-Based Collaborative Filtering (`solution/user_collaborative.py`)

**New Approach**: Recommend based on similar outsource companies.

**User Features**:
- **Historical preferences**: Industry/service distributions (TF-IDF weighted)
- **Client size profile**: Distribution over size buckets
- **Company profile**: Embedded description and services
- **Location preferences**: Geographic patterns

**How it works**:
```python
1. Build feature vectors for all users
2. Find k most similar users (cosine similarity)
3. Aggregate their triplet preferences
4. Weight by similarity scores
```

**Example**:
```
Company A: [Mobile Dev specialist, worked with 10 Healthcare startups]
Company B: [Mobile Dev specialist, worked with 12 FinTech startups]
→ High similarity (0.85)
→ Recommend FinTech leads to Company A
```

---

### 3. Triplet Content-Based (`solution/triplet_recommender.py`)

**Enhanced Content Matching** with multiple embedding strategies:

**Feature Components** (with weights):
- `triplet_structure` (30%): One-hot encodings of industry, size, services
- `background_text` (30%): Sentence-BERT embeddings of project descriptions
- `services_text` (20%): Embeddings of detailed service descriptions
- `location` (10%): Geographic features
- `numerical` (10%): Client size, budget statistics

**User Profile**: Mean pooling of historical interaction embeddings

---

### 4. Evaluation (`benchmark_data.py`)

**Two Evaluation Modes**:

#### A. Exact Match
```python
Predicted: "Healthcare|||medium|||Mobile,AI"
Ground Truth: "Healthcare|||medium|||Mobile,AI"
→ Hit = 1.0 ✅
```

#### B. Partial Match (Similarity-based)
```python
Predicted: "Healthcare|||medium|||Mobile,Web"
Ground Truth: "Healthcare|||large|||Mobile,AI"
→ Industry match: 1.0 (50% weight)
→ Size match: 0.5 (30% weight, adjacent buckets)
→ Services overlap: 0.5 (20% weight, 1/2 match)
→ Hit = 0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.5 = 0.75 ✅
```

**Metrics Reported**:
- Precision@K, Recall@K, F1@K
- MAP@K (Mean Average Precision)
- nDCG@K (Normalized Discounted Cumulative Gain)
- HitRate@K

**Both modes reported for comprehensive evaluation.**

---

## 🚀 Usage

### Quick Start

```bash
cd /home/ubuntu/crawl/crawler-recommend-sys/src
python execute_triplet.py
```

### Step-by-Step Workflow

```python
from triplet_utils import TripletManager, add_triplet_column
from solution.user_collaborative import UserBasedCollaborativeRecommender
from solution.triplet_recommender import TripletContentRecommender

# 1. Prepare data with triplets
triplet_manager = TripletManager(max_services=3)
triplet_manager.fit(df_train)  # Only on training data!

df_train = add_triplet_column(df_train, triplet_manager)
df_test = add_triplet_column(df_test, triplet_manager)

# 2. Train Content-Based
content_rec = TripletContentRecommender(
    df_history=df_train,
    df_test=df_test,
    triplet_manager=triplet_manager
)

# 3. Train Collaborative
collab_rec = UserBasedCollaborativeRecommender()
collab_rec.fit(df_train)

# 4. Get recommendations
content_recs = content_rec.recommend_triplets(user_id, top_k=10)
collab_recs = collab_rec.recommend_triplets(user_id, top_k=10)

# 5. Evaluate
from benchmark_data import BenchmarkOutput

# Exact match
benchmark = BenchmarkOutput(results_df, df_test)
summary, per_user = benchmark.evaluate_topk(k=10, use_partial_match=False)

# Partial match
def similarity_fn(t1, t2):
    return triplet_manager.calculate_triplet_similarity(t1, t2)

benchmark_partial = BenchmarkOutput(results_df, df_test, similarity_fn)
summary, per_user = benchmark_partial.evaluate_topk(
    k=10, 
    use_partial_match=True,
    partial_match_threshold=0.5
)
```

---

## 📊 Expected Results

### Performance Improvements

**Exact Match** (stricter):
- More specific recommendations
- Lower initial scores but higher precision
- Better for well-defined use cases

**Partial Match** (flexible):
- Captures "close enough" recommendations
- Higher recall
- Useful for discovery and exploration

### Ensemble Benefits

**Hybrid (70% Content + 30% Collaborative)**:
- Content-based: Generalizes to new triplets
- Collaborative: Learns from similar companies
- Combined: Best of both worlds

---

## 🔧 Configuration

### Triplet Weights
```python
# In triplet_manager.calculate_triplet_similarity()
weights = (
    0.5,  # industry (most important)
    0.3,  # client_size
    0.2   # services
)
```

### Embedding Weights
```python
# In TripletContentRecommender
embedding_weights = {
    'triplet_structure': 0.3,
    'background_text': 0.3,
    'services_text': 0.2,
    'location': 0.1,
    'numerical': 0.1
}
```

### Ensemble Weights
```python
# In hybrid ensemble
weights = (
    0.7,  # Content-based
    0.3   # Collaborative (user-user)
)
```

---

## 📁 Output Files

All results saved to: `/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/`

### Exact Match Results:
- `triplet_content_exact.csv` - Content-based summary
- `user_collab_exact.csv` - Collaborative summary
- `hybrid_ensemble_exact.csv` - Ensemble summary
- `*_per_user_exact.csv` - Per-user detailed metrics

### Partial Match Results:
- `triplet_content_partial.csv`
- `user_collab_partial.csv`
- `hybrid_ensemble_partial.csv`
- `*_per_user_partial.csv`

---

## 🎯 Key Advantages

1. **No Data Leakage**: 
   - TripletManager fitted only on training data
   - Test set never used during feature learning

2. **Dual Evaluation**:
   - Exact match: Strict, high precision
   - Partial match: Flexible, better recall

3. **User-User Learning**:
   - Leverages company similarity
   - Discovers hidden patterns
   - Better cold-start handling

4. **Actionable Recommendations**:
   - Specific industry + size + services
   - Clear targeting strategy
   - Reduced wasted effort

5. **Modular Design**:
   - Easy to add new recommenders
   - Flexible ensemble weights
   - Configurable similarity functions

---

## 🐛 Troubleshooting

### Issue: Low exact match scores
**Solution**: This is expected! Triplets are much more specific than single industries. Check partial match scores for better understanding.

### Issue: Sentence transformers not loading
**Solution**: Install required packages:
```bash
pip install sentence-transformers torch
```

### Issue: Memory errors
**Solution**: Reduce batch sizes in embedding functions or use smaller sentence transformer models.

---

## 🔮 Future Improvements

1. **Deep Learning Ensemble**: Neural network to learn optimal weights
2. **Time-aware Features**: Temporal patterns in client preferences
3. **Graph-based Methods**: Company-lead interaction graphs
4. **Active Learning**: User feedback to refine triplet similarity

---

## 📚 References

- Original industry-based approach: `src/solution/content_base_for_item.py`
- Collaborative filtering: `src/solution/collborative_for_item.py`
- Benchmark system: `src/benchmark_data.py`
