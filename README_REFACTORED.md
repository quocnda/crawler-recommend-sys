# Triplet-based Recommendation System

## Cấu trúc đã Refactor

```
src/
├── models/                        # Định nghĩa types & configs
│   ├── __init__.py               # Export all types
│   ├── types.py                  # Type definitions (Enums, Dataclasses)
│   └── config.py                 # Configuration classes
│
├── solution/                      # Các algorithms
│   ├── __init__.py               # Export all recommenders
│   ├── base_recommender.py       # Abstract base class (NEW)
│   ├── triplet_recommender.py    # Content-based recommender
│   ├── enhanced_triplet_content.py   # Enhanced content-based
│   ├── user_collaborative.py     # User-based collaborative
│   ├── enhanced_user_collaborative.py # Enhanced collaborative
│   ├── triplet_ensemble.py       # Ensemble methods
│   └── openai_embedder.py        # OpenAI embedding wrapper
│
├── preprocessing_data.py         # Data loading & preprocessing
├── triplet_utils.py              # Triplet creation & management
├── benchmark_data.py             # Evaluation metrics
├── execute_triplet_refactored.py # Main pipeline (NEW)
└── execute_triplet.py            # Original (deprecated)
```

## Các thay đổi chính

### 1. Type Definitions (`models/types.py`)

```python
# Enums
class EmbeddingType(Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"

class RecommenderMode(Enum):
    TRAIN = "train"
    TEST = "test"

class ClientSizeBucket(Enum):
    MICRO = (0, 10)
    SMALL = (11, 50)
    ...

# Type Aliases
UserID = str
TripletID = str
Score = float

# Dataclasses
@dataclass
class Triplet:
    industry: str
    client_size_bucket: str
    services: List[str]
    triplet_id: str

@dataclass
class RecommendationResult:
    user_id: UserID
    triplet_id: TripletID
    score: Score
    rank: int
```

### 2. Configuration (`models/config.py`)

```python
@dataclass
class EmbeddingConfig:
    embedding_type: EmbeddingType
    model_name: str
    embedding_dim: int

@dataclass
class PipelineConfig:
    train_path: Path
    test_path: Path
    embedding: EmbeddingConfig
    top_k: int
    experiments: List[str]
```

### 3. Base Recommender (`solution/base_recommender.py`)

```python
class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, df_train: pd.DataFrame, **kwargs) -> "BaseRecommender":
        pass
    
    @abstractmethod
    def recommend_triplets(
        self, 
        user_id: UserID, 
        top_k: int = 10, 
        **kwargs
    ) -> pd.DataFrame:
        pass
```

### 4. Main Pipeline (`execute_triplet_refactored.py`)

```python
# Configuration
config = PipelineConfig(
    train_path=Path("data/train.csv"),
    test_path=Path("data/test.csv"),
    embedding=EmbeddingSettings(use_openai=True),
    experiment=ExperimentSettings(top_k=10),
    experiments_to_run=[
        ExperimentType.TRIPLET_CONTENT,
        ExperimentType.ENHANCED_CONTENT,
    ]
)

# Run pipeline
results = run_pipeline(config)
```

## Sử dụng

### Chạy pipeline cơ bản

```python
from src.execute_triplet_refactored import run_pipeline, PipelineConfig

# Default config
results = run_pipeline()

# Custom config
config = PipelineConfig.default()
config.experiment.top_k = 20
config.experiments_to_run = [ExperimentType.TRIPLET_CONTENT]
results = run_pipeline(config)
```

### Sử dụng recommender riêng lẻ

```python
from src.solution import TripletContentRecommender
from src.preprocessing_data import preprocess_data
from src.triplet_utils import TripletManager, add_triplet_column

# Load data
df_train = preprocess_data("data/train.csv")

# Create triplets
manager = TripletManager()
manager.fit(df_train)
df_train = add_triplet_column(df_train, manager)

# Build recommender
recommender = TripletContentRecommender(
    df_history=df_train,
    df_test=df_test,
    triplet_manager=manager,
    use_openai=True
)

# Get recommendations
recs = recommender.recommend_triplets("user_123", top_k=10)
```

## Input/Output

### Input

**Training Data CSV:**
- `linkedin_company_outsource`: User/company identifier
- `industry`: Industry category
- `services`: Pipe-separated list of services
- `client_min`, `client_max`: Client size range
- `background`: Company description
- `location`: Company location

### Output

**Recommendations CSV:**
```
linkedin_company_outsource,triplet,score
company_123,IT|small|web_dev;mobile,0.85
company_123,Finance|medium|consulting,0.72
```

**Evaluation Metrics:**
- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- HitRate@K

## Experiments

| Experiment | Description |
|------------|-------------|
| `triplet_content` | Basic content-based with triplet features |
| `enhanced_content` | Multi-modal embeddings + industry hierarchy |
| `user_collaborative` | User-user collaborative filtering |
| `enhanced_collaborative` | Profile + history similarity |
| `triplet_ensemble` | Gradient Boosting meta-learner |
| `hybrid_ensemble` | Weighted combination |

## Dependencies

```
numpy
pandas
scikit-learn
sentence-transformers
openai
```

## Author

Quoc Nguyen - Version 2.0.0
