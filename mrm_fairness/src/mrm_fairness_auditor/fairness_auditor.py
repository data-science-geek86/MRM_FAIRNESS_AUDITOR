import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Any, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# 1. BINNING STRATEGIES
# =====================================================================

class BinningStrategy:
    """Abstract/Base interface for binning strategies."""
    @staticmethod
    def compute_edges(data: np.ndarray, num_bins: int, target: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

class EqualWidthBinning(BinningStrategy):
    @staticmethod
    def compute_edges(data: np.ndarray, num_bins: int, target: np.ndarray = None) -> np.ndarray:
        return np.linspace(data.min(), data.max(), num_bins + 1)

class EqualFrequencyBinning(BinningStrategy):
    @staticmethod
    def compute_edges(data: np.ndarray, num_bins: int, target: np.ndarray = None) -> np.ndarray:
        quantiles = np.linspace(0, 1, num_bins + 1)
        ## 26 June
        edges = np.percentile(data, quantiles * 100)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

class KMeansBinning(BinningStrategy):
    @staticmethod
    def compute_edges(data: np.ndarray, num_bins: int, target: np.ndarray = None) -> np.ndarray:
        if len(np.unique(data)) <= num_bins:
            return np.sort(np.unique(data))
        
        kmeans = KMeans(n_clusters=num_bins, random_state=42, n_init='auto')
        kmeans.fit(data.reshape(-1, 1))
        centroids = np.sort(kmeans.cluster_centers_.flatten())
        
        # Calculate midpoints between centroids to act as boundary edges
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        return np.array([-np.inf, *midpoints, np.inf])

class DecisionTreeBinning(BinningStrategy):
    @staticmethod
    def compute_edges(data: np.ndarray, num_bins: int, target: np.ndarray = None) -> np.ndarray:
        """Adaptive binning maximizing mutual information with a target label."""
        if target is None or len(np.unique(target)) < 2:
            # Fallback to equal frequency if a valid target vector isn't provided
            return EqualFrequencyBinning.compute_edges(data, num_bins)
        
        dt = DecisionTreeClassifier(max_leaf_nodes=num_bins, random_state=42)
        dt.fit(data.reshape(-1, 1), target)
        
        # Extract thresholds used by the splits
        thresholds = dt.tree_.threshold[dt.tree_.feature != -2]
        return np.sort(np.array([data.min(), *thresholds, data.max()]))


# Map string aliases to class definitions
BINNING_MAPPING = {
    'equal_width': EqualWidthBinning,
    'equal_frequency': EqualFrequencyBinning,
    'kmeans': KMeansBinning,
    'decision_tree': DecisionTreeBinning
}

# =====================================================================
# 2. CORE BIAS EVALUATION ENGINE
# =====================================================================

class BiasAuditor:
    def __init__(self, df: pd.DataFrame, protected_attribute: str, reference_group: Any, epsilon: float = 1e-4):
        self.df = df.copy()
        self.protected_attribute = protected_attribute
        self.reference_group = reference_group
        self.epsilon = epsilon
        self._validate_inputs()
        
    def _validate_inputs(self):
        if self.protected_attribute not in self.df.columns:
            raise ValueError(f"Column '{self.protected_attribute}' not found.")
        if self.reference_group not in self.df[self.protected_attribute].unique():
            raise ValueError(f"Reference group '{self.reference_group}' missing from dataset.")

    def calculate_psi(
        self, 
        target_cols: List[str], 
        binning_method: str = 'equal_frequency', 
        num_bins: int = 10,
        dt_target_col: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates Global PSI and outputs Granular Bin-Level Tables.
        """
        strategy = BINNING_MAPPING.get(binning_method.lower())
        if not strategy:
            raise ValueError(f"Invalid method. Choose from {list(BINNING_MAPPING.keys())}")
            
        summary_results = []
        granular_results = []
        
        ref_df = self.df[self.df[self.protected_attribute] == self.reference_group]
        comp_groups = [g for g in self.df[self.protected_attribute].dropna().unique() if g != self.reference_group]
        
        for col in target_cols:
            if col not in self.df.columns:
                continue
                
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
            ref_data = ref_df[col].dropna().values
            if len(ref_data) == 0: 
                continue
            
            # --- Establish Bin Definitions ---
            if not is_numeric:
                # Categorical alignment
                all_cats = self.df[col].dropna().unique()
                bin_labels = [str(cat) for cat in all_cats]
            else:
                # Continuous binning matching selection pattern
                target_y = ref_df[dt_target_col].values if dt_target_col else None
                edges = strategy.compute_edges(ref_data, num_bins, target=target_y)
                edges = np.unique(edges)
                if len(edges) < 2:
                    edges = np.array([edges[0] - 1e-5, edges[0] + 1e-5])
                bin_labels = [f"[{edges[i]:.4f} : {edges[i+1]:.4f}]" for i in range(len(edges)-1)]

            # Evaluate against each comparison demographics pocket
            for comp_group in comp_groups:
                comp_data = self.df[self.df[self.protected_attribute] == comp_group][col].dropna().values
                if len(comp_data) == 0: 
                    continue
                
                # --- Map to distributions ---
                if not is_numeric:
                    ref_counts = pd.Series(ref_data).value_counts(normalize=False).reindex(all_cats, fill_value=0)
                    comp_counts = pd.Series(comp_data).value_counts(normalize=False).reindex(all_cats, fill_value=0)
                else:
                    ref_counts, _ = np.histogram(ref_data, bins=edges)
                    comp_counts, _ = np.histogram(comp_data, bins=edges)
                
                # Compute probabilities
                ref_probs = ref_counts / len(ref_data)
                comp_probs = comp_counts / len(comp_data)
                
                # Apply anti-zero division adjustments
                ref_probs = np.where(ref_probs == 0, self.epsilon, ref_probs)
                comp_probs = np.where(comp_probs == 0, self.epsilon, comp_probs)
                ref_probs /= np.sum(ref_probs)
                comp_probs /= np.sum(comp_probs)
                
                # Element-wise contribution calculations
                bin_psi_contrib = (comp_probs - ref_probs) * np.log(comp_probs / ref_probs)
                total_psi = np.sum(bin_psi_contrib)
                
                # Granular Appending
                for idx, b_label in enumerate(bin_labels):
                    granular_results.append({
                        "Target Feature": col,
                        "Comparison Group": comp_group,
                        "Bin Range/Value": b_label,
                        "Reference Probability": ref_probs[idx],
                        "Comparison Probability": comp_probs[idx],
                        "PSI Contribution": bin_psi_contrib[idx]
                    })
                
                # Interpretation logic definitions
                if total_psi < 0.10:
                    status, action = "Fair / Minimal Disparity", "Acceptable stability profile."
                elif total_psi < 0.25:
                    status, action = "Mild Bias / Moderate Disparity", "Monitor carefully over time."
                else:
                    status, action = "Significant Bias / High Disparity", "Action Required! Mitigate bias dependencies."
                    
                summary_results.append({
                    "Protected Attribute": self.protected_attribute,
                    "Reference Group": self.reference_group,
                    "Comparison Group": comp_group,
                    "Target Feature/Prediction": col,
                    "PSI Value": total_psi,
                    "Fairness Status": status,
                    "Recommended Action": action
                })
                
        return pd.DataFrame(summary_results), pd.DataFrame(granular_results)

# =====================================================================
# 3. VISUALIZATION COMPONENT
# =====================================================================

class BiasVisualizer:
    @staticmethod
    def plot_granular_distribution(granular_df: pd.DataFrame, target_feature: str):
        """Generates clear interactive subplots contrasting distributions across bins."""
        filtered_df = granular_df[granular_df["Target Feature"] == target_feature]
        groups = filtered_df["Comparison Group"].unique()
        
        fig = make_subplots(rows=len(groups), cols=1, 
                            subplot_titles=[f"Reference vs Comparison ({g})" for g in groups],
                            shared_xaxes=True)
        
        for idx, g in enumerate(groups, 1):
            sub_df = filtered_df[filtered_df["Comparison Group"] == g]
            
            fig.add_trace(go.Bar(
                x=sub_df["Bin Range/Value"], y=sub_df["Reference Probability"],
                name="Reference Group", marker_color="#1f77b4", opacity=0.75,
                showlegend=(idx == 1)
            ), row=idx, col=1)
            
            fig.add_trace(go.Bar(
                x=sub_df["Bin Range/Value"], y=sub_df["Comparison Probability"],
                name=f"Comparison Group ({g})", marker_color="#ff7f0e", opacity=0.75,
                showlegend=(idx == 1)
            ), row=idx, col=1)
            
        fig.update_layout(
            title_text=f"Granular Bin Distribution Analysis for: {target_feature}",
            barmode='group', height=350 * len(groups), template="plotly_white"
        )
        fig.show()