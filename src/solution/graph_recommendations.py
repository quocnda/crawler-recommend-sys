"""
Graph-Based Recommendations using Network Analysis
=================================================

This module implements graph-based recommendation algorithms:
1. User-Item-Industry heterogeneous graph construction
2. Graph neural networks for recommendation
3. Random walk-based similarity
4. Community detection for user clustering
5. PageRank-based industry scoring
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, HeteroConv, Linear
    from torch_geometric.data import HeteroData
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    print("PyTorch Geometric not available. GNN features will be disabled.")


class HeterogeneousGraph:
    """
    Construct and analyze heterogeneous user-item-industry graph.
    """
    
    def __init__(self, alpha: float = 0.85, max_iter: int = 100):
        self.alpha = alpha  # PageRank damping factor
        self.max_iter = max_iter
        self.graph = nx.Graph()
        self.user_nodes = set()
        self.industry_nodes = set()
        self.service_nodes = set()
        
    def build_graph(self, df: pd.DataFrame):
        """
        Build heterogeneous graph from interaction data.
        """
        print("Building heterogeneous graph...")
        
        # Add nodes and edges
        for _, row in df.iterrows():
            user = f"user_{row['linkedin_company_outsource']}"
            industry = f"industry_{row['industry']}"
            location = f"location_{row.get('location', 'unknown')}"
            
            self.user_nodes.add(user)
            self.industry_nodes.add(industry)
            
            # User-Industry edges (primary interactions)
            self.graph.add_edge(user, industry, weight=1.0, edge_type='user_industry')
            
            # User-Location edges (geographic preference)
            self.graph.add_edge(user, location, weight=0.5, edge_type='user_location')
            
            # Industry-Location edges (industry distribution)
            self.graph.add_edge(industry, location, weight=0.3, edge_type='industry_location')
            
            # Service-based connections
            services = str(row.get('services', '')).split()
            for service in services[:5]:  # Limit to top 5 services
                service_node = f"service_{service.lower().strip()}"
                if len(service.strip()) > 2:  # Only meaningful services
                    self.service_nodes.add(service_node)
                    self.graph.add_edge(industry, service_node, weight=0.4, edge_type='industry_service')
                    self.graph.add_edge(user, service_node, weight=0.2, edge_type='user_service')
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
    def compute_pagerank_scores(self, personalization: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute PageRank scores for all nodes.
        """
        try:
            pagerank_scores = nx.pagerank(
                self.graph, 
                alpha=self.alpha,
                max_iter=self.max_iter,
                personalization=personalization
            )
            return pagerank_scores
        except:
            # Fallback to simple degree centrality
            return nx.degree_centrality(self.graph)
    
    def get_user_personalized_scores(self, user_id: str) -> Dict[str, float]:
        """
        Get personalized PageRank scores for a specific user.
        """
        user_node = f"user_{user_id}"
        if user_node not in self.graph:
            return {}
        
        # Create personalization vector (start random walk from user)
        personalization = {node: 0.0 for node in self.graph.nodes()}
        personalization[user_node] = 1.0
        
        scores = self.compute_pagerank_scores(personalization)
        
        # Filter to get industry scores
        industry_scores = {
            node.replace('industry_', ''): score 
            for node, score in scores.items() 
            if node.startswith('industry_')
        }
        
        return industry_scores
    
    def random_walk_similarity(self, user_id: str, walk_length: int = 50, num_walks: int = 100) -> Dict[str, float]:
        """
        Compute similarity using random walks from user node.
        """
        user_node = f"user_{user_id}"
        if user_node not in self.graph:
            return {}
        
        industry_visits = Counter()
        
        for _ in range(num_walks):
            current_node = user_node
            
            for _ in range(walk_length):
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    break
                
                # Weighted random selection
                weights = [self.graph[current_node][neighbor].get('weight', 1.0) for neighbor in neighbors]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    probs = [w/total_weight for w in weights]
                    current_node = np.random.choice(neighbors, p=probs)
                else:
                    current_node = np.random.choice(neighbors)
                
                # Count industry visits
                if current_node.startswith('industry_'):
                    industry_visits[current_node.replace('industry_', '')] += 1
        
        # Normalize counts
        total_visits = sum(industry_visits.values())
        if total_visits > 0:
            return {industry: count/total_visits for industry, count in industry_visits.items()}
        return {}
    
    def detect_communities(self) -> Dict[str, int]:
        """
        Detect communities in the graph using Louvain algorithm.
        """
        try:
            communities = nx.community.louvain_communities(self.graph)
            node_to_community = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = i
            return node_to_community
        except:
            # Fallback: assign all nodes to community 0
            return {node: 0 for node in self.graph.nodes()}
    
    def get_similar_users(self, user_id: str, top_k: int = 20) -> List[str]:
        """
        Find similar users based on graph structure.
        """
        user_node = f"user_{user_id}"
        if user_node not in self.graph:
            return []
        
        # Get user's neighbors (industries they interact with)
        user_neighbors = set(self.graph.neighbors(user_node))
        
        similarities = {}
        for other_user in self.user_nodes:
            if other_user == user_node:
                continue
            
            other_neighbors = set(self.graph.neighbors(other_user))
            
            # Jaccard similarity
            intersection = len(user_neighbors & other_neighbors)
            union = len(user_neighbors | other_neighbors)
            
            if union > 0:
                similarities[other_user.replace('user_', '')] = intersection / union
        
        # Return top similar users
        sorted_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [user for user, _ in sorted_users[:top_k]]


class GraphNeuralRecommender(nn.Module):
    """
    Graph Neural Network for recommendation using heterogeneous graphs.
    """
    
    def __init__(
        self,
        node_types: Dict[str, int],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        
        if not TORCH_GEO_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GraphNeuralRecommender")
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        
        # Node embeddings
        self.embeddings = nn.ModuleDict()
        for node_type, num_nodes in node_types.items():
            self.embeddings[node_type] = nn.Embedding(num_nodes, hidden_dim)
        
        # Heterogeneous graph convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict))
        
        # Output layer
        self.output = Linear(hidden_dim, 1)
        
    def forward(self, x_dict, edge_index_dict):
        # Apply embeddings
        for node_type in x_dict:
            x_dict[node_type] = self.embeddings[node_type](x_dict[node_type])
        
        # Apply graph convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        return x_dict


class GraphBasedRecommender:
    """
    Complete graph-based recommendation system.
    """
    
    def __init__(
        self,
        use_pagerank: bool = True,
        use_random_walk: bool = True,
        use_community: bool = True,
        use_gnn: bool = False,
        random_walk_params: Optional[Dict] = None
    ):
        self.use_pagerank = use_pagerank
        self.use_random_walk = use_random_walk
        self.use_community = use_community
        self.use_gnn = use_gnn and TORCH_GEO_AVAILABLE
        
        self.random_walk_params = random_walk_params or {
            'walk_length': 30,
            'num_walks': 50
        }
        
        self.graph = HeterogeneousGraph()
        self.communities = {}
        self.gnn_model = None
        self.industry_popularity = {}
        
    def fit(self, df_history: pd.DataFrame):
        """
        Fit the graph-based recommender.
        """
        print("Fitting Graph-Based Recommender...")
        
        # Build graph
        self.graph.build_graph(df_history)
        
        # Compute industry popularity
        industry_counts = df_history['industry'].value_counts()
        total_interactions = len(df_history)
        self.industry_popularity = {
            industry: count / total_interactions
            for industry, count in industry_counts.items()
        }
        
        # Detect communities
        if self.use_community:
            print("Detecting communities...")
            self.communities = self.graph.detect_communities()
        
        # Initialize GNN if requested
        if self.use_gnn:
            try:
                self._initialize_gnn(df_history)
            except Exception as e:
                print(f"GNN initialization failed: {e}")
                self.use_gnn = False
        
        print("Graph-based recommender fitting completed!")
    
    def _initialize_gnn(self, df_history: pd.DataFrame):
        """
        Initialize and train Graph Neural Network.
        """
        if not TORCH_GEO_AVAILABLE:
            return
        
        # This is a simplified GNN setup - in practice would need more sophisticated data preparation
        print("Initializing GNN (simplified version)...")
        
        # Create node mappings
        users = df_history['linkedin_company_outsource'].unique()
        industries = df_history['industry'].unique()
        
        node_types = {
            'user': len(users),
            'industry': len(industries)
        }
        
        edge_types = [
            ('user', 'interacts', 'industry'),
            ('industry', 'rev_interacts', 'user')
        ]
        
        self.gnn_model = GraphNeuralRecommender(node_types, edge_types)
        print("GNN initialized (training would require more complex data preparation)")
    
    def recommend_items(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations using graph-based methods.
        """
        scores = defaultdict(float)
        
        # Method 1: Personalized PageRank
        if self.use_pagerank:
            try:
                pagerank_scores = self.graph.get_user_personalized_scores(user_id)
                for industry, score in pagerank_scores.items():
                    scores[industry] += 0.4 * score
            except Exception as e:
                print(f"PageRank failed: {e}")
        
        # Method 2: Random Walk Similarity
        if self.use_random_walk:
            try:
                rw_scores = self.graph.random_walk_similarity(user_id, **self.random_walk_params)
                for industry, score in rw_scores.items():
                    scores[industry] += 0.3 * score
            except Exception as e:
                print(f"Random walk failed: {e}")
        
        # Method 3: Community-based Recommendations
        if self.use_community and self.communities:
            try:
                community_scores = self._get_community_recommendations(user_id)
                for industry, score in community_scores.items():
                    scores[industry] += 0.2 * score
            except Exception as e:
                print(f"Community recommendations failed: {e}")
        
        # Method 4: Similar Users (Graph-based CF)
        try:
            similar_user_scores = self._get_similar_user_recommendations(user_id)
            for industry, score in similar_user_scores.items():
                scores[industry] += 0.1 * score
        except Exception as e:
            print(f"Similar user recommendations failed: {e}")
        
        # Fallback: Use popularity if no scores
        if not scores:
            scores = self.industry_popularity.copy()
        
        # Sort and return top recommendations
        sorted_industries = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = pd.DataFrame([
            {'industry': industry, 'score': score}
            for industry, score in sorted_industries
        ])
        
        return results
    
    def _get_community_recommendations(self, user_id: str) -> Dict[str, float]:
        """
        Get recommendations based on community membership.
        """
        user_node = f"user_{user_id}"
        if user_node not in self.communities:
            return {}
        
        user_community = self.communities[user_node]
        
        # Find industries in the same community
        community_industries = []
        for node, community in self.communities.items():
            if community == user_community and node.startswith('industry_'):
                community_industries.append(node.replace('industry_', ''))
        
        # Score based on popularity within community
        scores = {}
        for industry in community_industries:
            scores[industry] = self.industry_popularity.get(industry, 0.001)
        
        return scores
    
    def _get_similar_user_recommendations(self, user_id: str) -> Dict[str, float]:
        """
        Get recommendations from similar users in the graph.
        """
        similar_users = self.graph.get_similar_users(user_id, top_k=10)
        
        if not similar_users:
            return {}
        
        # Aggregate industries from similar users
        industry_scores = defaultdict(float)
        
        for similar_user in similar_users:
            similar_user_node = f"user_{similar_user}"
            if similar_user_node in self.graph.graph:
                # Get industries this similar user interacts with
                for neighbor in self.graph.graph.neighbors(similar_user_node):
                    if neighbor.startswith('industry_'):
                        industry = neighbor.replace('industry_', '')
                        weight = self.graph.graph[similar_user_node][neighbor].get('weight', 1.0)
                        industry_scores[industry] += weight
        
        # Normalize scores
        total_score = sum(industry_scores.values())
        if total_score > 0:
            return {industry: score/total_score for industry, score in industry_scores.items()}
        
        return {}
    
    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the constructed graph.
        """
        stats = {
            'num_nodes': self.graph.graph.number_of_nodes(),
            'num_edges': self.graph.graph.number_of_edges(),
            'num_users': len(self.graph.user_nodes),
            'num_industries': len(self.graph.industry_nodes),
            'num_services': len(self.graph.service_nodes),
            'avg_degree': np.mean([self.graph.graph.degree(node) for node in self.graph.graph.nodes()]),
            'density': nx.density(self.graph.graph)
        }
        
        if self.communities:
            stats['num_communities'] = len(set(self.communities.values()))
        
        return stats


def integrate_graph_recommendations(
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    top_k: int = 10,
    graph_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Integration function for graph-based recommendations.
    """
    config = graph_config or {
        'use_pagerank': True,
        'use_random_walk': True,
        'use_community': True,
        'use_gnn': False,  # Disabled by default due to complexity
        'random_walk_params': {'walk_length': 20, 'num_walks': 30}
    }
    
    print("Initializing Graph-Based Recommender...")
    recommender = GraphBasedRecommender(**config)
    
    # Fit on historical data
    recommender.fit(df_history)
    
    # Print graph statistics
    stats = recommender.get_graph_statistics()
    print(f"Graph Statistics: {stats}")
    
    # Generate recommendations for test users
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id) or user_id in seen_users:
            continue
        seen_users.add(user_id)
        
        try:
            recs = recommender.recommend_items(user_id, top_k=top_k)
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'industry': rec['industry'],
                    'score': rec['score']
                })
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue
    
    return pd.DataFrame(results)