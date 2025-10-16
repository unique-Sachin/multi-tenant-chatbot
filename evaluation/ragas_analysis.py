"""
Jupyter notebook for interactive RAGAS evaluation and analysis.
Run this after installing: pip install -r requirements-dev.txt
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import asyncio
from pathlib import Path

# Import our evaluation framework
import sys
sys.path.append('.')
from ragas_evaluator import RAGASEvaluator

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your deployed URL
AUTH_TOKEN = None  # Set your JWT token if needed

class RAGASAnalysis:
    """Interactive analysis tools for RAGAS results."""
    
    def __init__(self):
        self.evaluator = RAGASEvaluator(API_BASE_URL, AUTH_TOKEN)
    
    def load_results(self, results_file: str) -> dict:
        """Load evaluation results from JSON file."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def plot_metrics_comparison(self, results: dict, title: str = "RAGAS Metrics Comparison"):
        """Create interactive plot comparing metrics across namespaces or methods."""
        
        # Extract metrics data
        plot_data = []
        
        if "results" in results:  # Comparison results
            for method, result in results["results"].items():
                if "metrics" in result:
                    for metric, score in result["metrics"].items():
                        if isinstance(score, (int, float)):
                            plot_data.append({
                                "Method": method,
                                "Metric": metric.replace('_', ' ').title(),
                                "Score": score
                            })
        else:  # Single result or multiple namespaces
            if "metrics" in results:  # Single namespace
                namespace = results.get("namespace", "Unknown")
                for metric, score in results["metrics"].items():
                    if isinstance(score, (int, float)):
                        plot_data.append({
                            "Namespace": namespace,
                            "Metric": metric.replace('_', ' ').title(), 
                            "Score": score
                        })
            else:  # Multiple namespaces
                for namespace, result in results.items():
                    if isinstance(result, dict) and "metrics" in result:
                        for metric, score in result["metrics"].items():
                            if isinstance(score, (int, float)):
                                plot_data.append({
                                    "Namespace": namespace,
                                    "Metric": metric.replace('_', ' ').title(),
                                    "Score": score
                                })
        
        if not plot_data:
            print("No metrics data found to plot")
            return None
        
        df = pd.DataFrame(plot_data)
        
        # Determine grouping column
        group_col = "Method" if "Method" in df.columns else "Namespace"
        
        # Create interactive bar chart
        fig = px.bar(
            df, 
            x="Metric", 
            y="Score", 
            color=group_col,
            title=title,
            labels={"Score": "Score (0-1)", "Metric": "RAGAS Metrics"},
            height=600
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.show()
        return fig
    
    def plot_radar_chart(self, results: dict, title: str = "RAGAS Metrics Radar"):
        """Create radar chart for metrics visualization."""
        
        # Extract metrics for radar chart
        if "metrics" in results:  # Single result
            metrics = results["metrics"]
            categories = [metric.replace('_', ' ').title() for metric in metrics.keys() 
                         if isinstance(metrics[metric], (int, float))]
            values = [score for score in metrics.values() 
                     if isinstance(score, (int, float))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=results.get("namespace", "Evaluation")
            ))
            
        else:  # Multiple results
            fig = go.Figure()
            
            for name, result in results.items():
                if isinstance(result, dict) and "metrics" in result:
                    metrics = result["metrics"]
                    categories = [metric.replace('_', ' ').title() for metric in metrics.keys() 
                                 if isinstance(metrics[metric], (int, float))]
                    values = [score for score in metrics.values() 
                             if isinstance(score, (int, float))]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=name
                    ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title,
            template="plotly_white"
        )
        
        fig.show()
        return fig
    
    def analyze_detailed_results(self, results: dict) -> pd.DataFrame:
        """Analyze detailed question-level results."""
        
        detailed_data = []
        
        if "detailed_results" in results:
            for i, item in enumerate(results["detailed_results"]):
                detailed_data.append({
                    "Question_ID": i + 1,
                    "Question": item.get("question", "")[:50] + "...",
                    "Answer_Length": len(item.get("answer", "")),
                    "Context_Count": len(item.get("contexts", [])),
                    "Has_Ground_Truth": bool(item.get("ground_truth", "")),
                    "Is_Out_of_Scope": "out of scope" in item.get("answer", "").lower()
                })
        
        return pd.DataFrame(detailed_data)
    
    def generate_summary_report(self, results: dict) -> dict:
        """Generate summary statistics."""
        
        summary = {
            "total_evaluations": 0,
            "avg_scores": {},
            "min_scores": {},
            "max_scores": {},
            "score_distribution": {}
        }
        
        all_scores = {}
        
        # Collect all scores
        if "metrics" in results:  # Single result
            for metric, score in results["metrics"].items():
                if isinstance(score, (int, float)):
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)
            summary["total_evaluations"] = 1
            
        else:  # Multiple results
            for name, result in results.items():
                if isinstance(result, dict) and "metrics" in result:
                    for metric, score in result["metrics"].items():
                        if isinstance(score, (int, float)):
                            if metric not in all_scores:
                                all_scores[metric] = []
                            all_scores[metric].append(score)
                    summary["total_evaluations"] += 1
        
        # Calculate statistics
        for metric, scores in all_scores.items():
            summary["avg_scores"][metric] = sum(scores) / len(scores)
            summary["min_scores"][metric] = min(scores)
            summary["max_scores"][metric] = max(scores)
            
            # Score distribution
            if len(scores) > 1:
                summary["score_distribution"][metric] = {
                    "std": pd.Series(scores).std(),
                    "range": max(scores) - min(scores)
                }
        
        return summary

# Example usage functions for the notebook
async def run_quick_evaluation(namespace: str = "zibtek"):
    """Run a quick evaluation for testing."""
    
    test_cases = [
        {
            "question": "What services does Zibtek offer?",
            "ground_truth": "Zibtek offers software development services."
        },
        {
            "question": "How can I contact Zibtek?",
            "ground_truth": "You can contact Zibtek through their website."
        },
        {
            "question": "What is the weather today?",
            "ground_truth": "This should be out of scope."
        }
    ]
    
    evaluator = RAGASEvaluator(API_BASE_URL, AUTH_TOKEN)
    results = await evaluator.evaluate_test_set(test_cases, namespace)
    
    return results

def analyze_results_from_file(results_file: str):
    """Analyze results from a saved JSON file."""
    
    analyzer = RAGASAnalysis()
    results = analyzer.load_results(results_file)
    
    print("📊 RAGAS Analysis Results")
    print("=" * 50)
    
    # Generate plots
    analyzer.plot_metrics_comparison(results)
    analyzer.plot_radar_chart(results)
    
    # Summary statistics
    summary = analyzer.generate_summary_report(results)
    print(f"\nTotal Evaluations: {summary['total_evaluations']}")
    print("\nAverage Scores:")
    for metric, score in summary["avg_scores"].items():
        print(f"  {metric}: {score:.3f}")
    
    # Detailed analysis
    if "detailed_results" in results:
        detailed_df = analyzer.analyze_detailed_results(results)
        print(f"\nDetailed Analysis:")
        print(detailed_df.describe())
        return detailed_df
    
    return summary

# Notebook helper functions
def display_sample_usage():
    """Display sample usage examples."""
    
    print("""
    🧪 RAGAS Evaluation Notebook
    ============================
    
    # 1. Quick evaluation
    results = await run_quick_evaluation("zibtek")
    
    # 2. Analyze saved results
    df = analyze_results_from_file("evaluation/results.json")
    
    # 3. Custom evaluation
    evaluator = RAGASEvaluator("http://localhost:8000", None)
    custom_results = await evaluator.evaluate_test_set(your_test_cases, "your_namespace")
    
    # 4. Interactive analysis
    analyzer = RAGASAnalysis()
    analyzer.plot_metrics_comparison(results)
    analyzer.plot_radar_chart(results)
    
    # 5. Compare multiple files
    results1 = analyzer.load_results("evaluation/result1.json")
    results2 = analyzer.load_results("evaluation/result2.json")
    combined = {"Method 1": results1, "Method 2": results2}
    analyzer.plot_metrics_comparison(combined, "Method Comparison")
    """)