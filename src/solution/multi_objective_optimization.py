"""
Multi-Objective Optimization for Recommendation System
====================================================

This module implements multi-objective optimization to balance:
1. Relevance (accuracy metrics)
2. Diversity (intra-list diversity)
3. Novelty (serendipity)  
4. Business metrics (revenue potential, strategic importance)
5. Coverage (catalog coverage)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("PyMOO not available. Multi-objective optimization will use simplified approach.")


class RecommendationObjectives:
    """
    Calculate various objectives for recommendation optimization.
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.3,
        novelty_weight: float = 0.2,
        business_weight: float = 0.2,
        coverage_weight: float = 0.1
    ):
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        self.business_weight = business_weight
        self.coverage_weight = coverage_weight
        
    def calculate_relevance(
        self,
        recommendations: List[str],
        base_scores: Dict[str, float]
    ) -> float:
        """
        Calculate relevance score based on base model predictions.
        """
        if not recommendations:
            return 0.0
        
        relevance_scores = [base_scores.get(item, 0.0) for item in recommendations]
        return np.mean(relevance_scores)
    
    def calculate_diversity(
        self,
        recommendations: List[str],
        item_features: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate intra-list diversity using feature-based similarity.
        """
        if len(recommendations) <= 1:
            return 0.0
        
        # Get feature vectors for recommendations
        feature_vectors = []
        for item in recommendations:
            if item in item_features:
                feature_vectors.append(item_features[item])
        
        if len(feature_vectors) <= 1:
            return 0.0
        
        # Calculate pairwise distances
        feature_matrix = np.array(feature_vectors)
        distances = pairwise_distances(feature_matrix, metric='cosine')
        
        # Average distance (diversity)
        n_items = len(feature_vectors)
        total_distance = 0
        count = 0
        
        for i in range(n_items):
            for j in range(i+1, n_items):
                total_distance += distances[i, j]
                count += 1
        
        return total_distance / max(count, 1)
    
    def calculate_novelty(
        self,
        recommendations: List[str],
        user_history: List[str],
        global_popularity: Dict[str, float]
    ) -> float:
        """
        Calculate novelty (serendipity) score.
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        
        for item in recommendations:
            # Novelty = 1 - popularity (less popular = more novel)
            popularity = global_popularity.get(item, 0.0)
            item_novelty = 1.0 - popularity
            
            # Boost novelty if item is not in user history
            if item not in user_history:
                item_novelty *= 1.2
            
            novelty_scores.append(item_novelty)
        
        return np.mean(novelty_scores)
    
    def calculate_business_value(
        self,
        recommendations: List[str],
        business_metrics: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calculate business value score.
        """
        if not recommendations:
            return 0.0
        
        business_scores = []
        
        for item in recommendations:
            item_metrics = business_metrics.get(item, {})
            
            # Business value components
            revenue_potential = item_metrics.get('revenue_potential', 0.5)
            strategic_importance = item_metrics.get('strategic_importance', 0.5)
            market_growth = item_metrics.get('market_growth', 0.5)
            
            # Combined business score
            business_score = (
                0.5 * revenue_potential +
                0.3 * strategic_importance +
                0.2 * market_growth
            )
            
            business_scores.append(business_score)
        
        return np.mean(business_scores)
    
    def calculate_coverage(
        self,
        recommendations: List[str],
        total_catalog: List[str]
    ) -> float:
        """
        Calculate catalog coverage contribution.
        """
        if not total_catalog:
            return 0.0
        
        unique_recommendations = set(recommendations)
        coverage = len(unique_recommendations) / len(total_catalog)
        return min(coverage, 1.0)


class MultiObjectiveProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for recommendations.
    """
    
    def __init__(
        self,
        candidate_items: List[str],
        base_scores: Dict[str, float],
        item_features: Dict[str, np.ndarray],
        user_history: List[str],
        global_popularity: Dict[str, float],
        business_metrics: Dict[str, Dict[str, float]],
        list_size: int = 10
    ):
        self.candidate_items = candidate_items
        self.base_scores = base_scores
        self.item_features = item_features
        self.user_history = user_history
        self.global_popularity = global_popularity
        self.business_metrics = business_metrics
        self.list_size = list_size
        
        self.objectives = RecommendationObjectives()
        
        # Problem definition: binary variables for each candidate item
        super().__init__(
            n_var=len(candidate_items),
            n_obj=4,  # relevance, diversity, novelty, business_value
            n_constr=1,  # constraint: exactly list_size items selected
            xl=0,  # lower bound (binary)
            xu=1   # upper bound (binary)
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate objectives for a given solution.
        """
        # Convert binary solution to recommendation list
        selected_indices = np.where(x > 0.5)[0]
        recommendations = [self.candidate_items[i] for i in selected_indices]
        
        # Calculate objectives
        relevance = self.objectives.calculate_relevance(recommendations, self.base_scores)
        diversity = self.objectives.calculate_diversity(recommendations, self.item_features)
        novelty = self.objectives.calculate_novelty(recommendations, self.user_history, self.global_popularity)
        business_value = self.objectives.calculate_business_value(recommendations, self.business_metrics)
        
        # Objectives to minimize (negative because we want to maximize)
        out["F"] = [-relevance, -diversity, -novelty, -business_value]
        
        # Constraint: exactly list_size items selected
        out["G"] = [abs(len(recommendations) - self.list_size)]


class ParetoBandit:
    """
    Multi-armed bandit approach for Pareto-optimal recommendations.
    """
    
    def __init__(
        self,
        n_objectives: int = 4,
        exploration_rate: float = 0.1,
        learning_rate: float = 0.01
    ):
        self.n_objectives = n_objectives
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        
        # Bandit state
        self.arm_counts = {}
        self.arm_rewards = {}
        self.pareto_solutions = []
        
    def select_arm(self, available_arms: List[str]) -> str:
        """
        Select arm using Upper Confidence Bound with Pareto dominance.
        """
        if not available_arms:
            return None
        
        # Exploration: select random arm
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_arms)
        
        # Exploitation: select based on Pareto dominance
        best_arm = None
        best_score = float('-inf')
        
        for arm in available_arms:
            if arm not in self.arm_rewards:
                return arm  # Explore new arm
            
            # Calculate UCB score for multi-objective
            avg_rewards = self.arm_rewards[arm]
            count = self.arm_counts[arm]
            
            # Multi-objective UCB
            confidence_bonus = np.sqrt(2 * np.log(sum(self.arm_counts.values())) / count)
            ucb_score = np.mean(avg_rewards) + confidence_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_arm = arm
        
        return best_arm or np.random.choice(available_arms)
    
    def update_arm(self, arm: str, rewards: np.ndarray):
        """
        Update arm statistics with multi-objective rewards.
        """
        if arm not in self.arm_rewards:
            self.arm_rewards[arm] = np.zeros(self.n_objectives)
            self.arm_counts[arm] = 0
        
        # Update running average
        self.arm_counts[arm] += 1
        alpha = self.learning_rate
        self.arm_rewards[arm] = (1 - alpha) * self.arm_rewards[arm] + alpha * rewards
    
    def is_pareto_dominated(self, solution1: np.ndarray, solution2: np.ndarray) -> bool:
        """
        Check if solution1 is Pareto dominated by solution2.
        """
        return np.all(solution1 <= solution2) and np.any(solution1 < solution2)
    
    def update_pareto_front(self, new_solution: Tuple[str, np.ndarray]):
        """
        Update Pareto front with new solution.
        """
        arm, rewards = new_solution
        
        # Remove dominated solutions
        self.pareto_solutions = [
            (existing_arm, existing_rewards)
            for existing_arm, existing_rewards in self.pareto_solutions
            if not self.is_pareto_dominated(existing_rewards, rewards)
        ]
        
        # Add new solution if not dominated
        dominated = any(
            self.is_pareto_dominated(rewards, existing_rewards)
            for _, existing_rewards in self.pareto_solutions
        )
        
        if not dominated:
            self.pareto_solutions.append((arm, rewards))


class MultiObjectiveRecommender:
    """
    Complete multi-objective recommendation system.
    """
    
    def __init__(
        self,
        use_pareto_optimization: bool = True,
        use_bandit_learning: bool = True,
        optimization_method: str = 'nsga2',
        bandit_config: Optional[Dict] = None
    ):
        self.use_pareto_optimization = use_pareto_optimization and PYMOO_AVAILABLE
        self.use_bandit_learning = use_bandit_learning
        self.optimization_method = optimization_method
        
        self.bandit_config = bandit_config or {
            'n_objectives': 4,
            'exploration_rate': 0.1,
            'learning_rate': 0.01
        }
        
        # Initialize components
        self.pareto_bandit = ParetoBandit(**self.bandit_config) if use_bandit_learning else None
        self.global_popularity = {}
        self.business_metrics = {}
        self.item_features = {}
        
    def fit(self, df_history: pd.DataFrame, df_candidates: pd.DataFrame):
        """
        Fit multi-objective recommender.
        """
        print("Fitting Multi-Objective Recommender...")
        
        # Calculate global popularity
        industry_counts = df_history['industry'].value_counts()
        total_interactions = len(df_history)
        self.global_popularity = {
            industry: count / total_interactions
            for industry, count in industry_counts.items()
        }
        
        # Generate business metrics (simplified - in practice would come from business data)
        unique_industries = df_candidates['industry'].unique()
        for industry in unique_industries:
            self.business_metrics[industry] = {
                'revenue_potential': np.random.uniform(0.3, 1.0),
                'strategic_importance': np.random.uniform(0.2, 0.9),
                'market_growth': np.random.uniform(0.1, 0.8)
            }
        
        # Generate item features for diversity calculation
        for industry in unique_industries:
            # Simple feature vector based on industry characteristics
            feature_vector = np.random.normal(0, 1, 20)  # 20-dim features
            # Add some structure based on industry name
            industry_hash = hash(industry) % 1000
            feature_vector[:5] = industry_hash / 1000.0
            self.item_features[industry] = feature_vector
        
        print("Multi-objective recommender fitting completed!")
    
    def optimize_recommendations(
        self,
        candidate_items: List[str],
        base_scores: Dict[str, float],
        user_history: List[str],
        list_size: int = 10
    ) -> List[str]:
        """
        Optimize recommendations using multi-objective approach.
        """
        if not self.use_pareto_optimization:
            return self._greedy_multi_objective(candidate_items, base_scores, user_history, list_size)
        
        try:
            # Create multi-objective problem
            problem = MultiObjectiveProblem(
                candidate_items=candidate_items,
                base_scores=base_scores,
                item_features=self.item_features,
                user_history=user_history,
                global_popularity=self.global_popularity,
                business_metrics=self.business_metrics,
                list_size=list_size
            )
            
            # Solve with NSGA-II
            algorithm = NSGA2(pop_size=100)
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', 50),
                verbose=False
            )
            
            # Select best solution from Pareto front
            if result.X is not None and len(result.X) > 0:
                # Choose solution with best compromise (closest to ideal point)
                objectives = -result.F  # Convert back to maximization
                
                # Normalize objectives
                obj_max = np.max(objectives, axis=0)
                obj_min = np.min(objectives, axis=0)
                normalized_obj = (objectives - obj_min) / (obj_max - obj_min + 1e-8)
                
                # Find solution closest to ideal point (1,1,1,1)
                distances = np.sqrt(np.sum((normalized_obj - 1.0) ** 2, axis=1))
                best_idx = np.argmin(distances)
                
                # Convert solution to recommendation list
                best_solution = result.X[best_idx]
                selected_indices = np.where(best_solution > 0.5)[0]
                recommendations = [candidate_items[i] for i in selected_indices]
                
                return recommendations[:list_size]
                
        except Exception as e:
            print(f"Pareto optimization failed: {e}")
            return self._greedy_multi_objective(candidate_items, base_scores, user_history, list_size)
    
    def _greedy_multi_objective(
        self,
        candidate_items: List[str],
        base_scores: Dict[str, float],
        user_history: List[str],
        list_size: int
    ) -> List[str]:
        """
        Greedy multi-objective optimization fallback.
        """
        objectives = RecommendationObjectives()
        selected_items = []
        remaining_items = candidate_items.copy()
        
        for _ in range(min(list_size, len(candidate_items))):
            best_item = None
            best_score = float('-inf')
            
            for item in remaining_items:
                # Calculate objectives for current selection + this item
                test_selection = selected_items + [item]
                
                relevance = objectives.calculate_relevance(test_selection, base_scores)
                diversity = objectives.calculate_diversity(test_selection, self.item_features)
                novelty = objectives.calculate_novelty(test_selection, user_history, self.global_popularity)
                business_value = objectives.calculate_business_value(test_selection, self.business_metrics)
                
                # Weighted combination
                combined_score = (
                    0.4 * relevance +
                    0.25 * diversity +
                    0.2 * novelty +
                    0.15 * business_value
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_item = item
            
            if best_item:
                selected_items.append(best_item)
                remaining_items.remove(best_item)
        
        return selected_items
    
    def recommend_items(
        self,
        user_id: str,
        base_predictions: Dict[str, float],
        user_history: List[str],
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Generate multi-objective optimized recommendations.
        """
        # Get candidate items
        candidate_items = list(base_predictions.keys())
        
        if not candidate_items:
            return pd.DataFrame(columns=['industry', 'score'])
        
        # Optimize recommendations
        optimized_recommendations = self.optimize_recommendations(
            candidate_items, base_predictions, user_history, top_k
        )
        
        # Update bandit learning if enabled
        if self.pareto_bandit:
            for item in optimized_recommendations:
                # Simulate multi-objective rewards (in practice, would come from user feedback)
                rewards = np.array([
                    base_predictions.get(item, 0.0),  # relevance
                    np.random.uniform(0.3, 0.9),      # diversity (simulated)
                    1.0 - self.global_popularity.get(item, 0.5),  # novelty
                    self.business_metrics.get(item, {}).get('revenue_potential', 0.5)  # business
                ])
                self.pareto_bandit.update_arm(item, rewards)
        
        # Create results DataFrame
        results = []
        for i, item in enumerate(optimized_recommendations):
            # Score based on position and base score
            position_penalty = 1.0 - (i * 0.05)  # Small penalty for lower positions
            final_score = base_predictions.get(item, 0.0) * position_penalty
            
            results.append({
                'industry': item,
                'score': final_score
            })
        
        return pd.DataFrame(results)


def integrate_multi_objective_optimization(
    base_predictions: Dict[str, Dict[str, float]],  # {user_id: {industry: score}}
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Integration function for multi-objective optimization.
    """
    print("Initializing Multi-Objective Recommender...")
    
    recommender = MultiObjectiveRecommender(
        use_pareto_optimization=PYMOO_AVAILABLE,
        use_bandit_learning=True
    )
    
    # Fit recommender
    recommender.fit(df_history, df_test)
    
    # Generate optimized recommendations
    results = []
    
    for user_id, user_predictions in base_predictions.items():
        # Get user history
        user_history = df_history[df_history['linkedin_company_outsource'] == user_id]['industry'].tolist()
        
        try:
            recs = recommender.recommend_items(user_id, user_predictions, user_history, top_k)
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