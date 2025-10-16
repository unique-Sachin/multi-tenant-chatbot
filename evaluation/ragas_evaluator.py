"""
RAGAS Evaluation Framework for Multi-tenant RAG System
Based on latest RAGAS documentation: https://docs.ragas.io
"""

import os
import asyncio
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import requests

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy, 
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
    context_utilization,
    context_entity_recall,
    summarization_score
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

class RAGASEvaluator:
    """RAGAS evaluation for multi-tenant RAG chatbot system."""
    
    def __init__(self, api_base_url: str = None, auth_token: str = None):
        """Initialize RAGAS evaluator.
        
        Args:
            api_base_url: Base URL of your deployed API
            auth_token: JWT token for authentication
        """
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.auth_token = auth_token or os.getenv("AUTH_TOKEN")
        
        # Initialize RAGAS models
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            timeout=60
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        print(f"🧪 RAGAS Evaluator initialized")
        print(f"   API URL: {self.api_base_url}")
        print(f"   Auth: {'✅ Configured' if self.auth_token else '❌ Missing'}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}
    
    async def get_rag_response(
        self, 
        question: str, 
        namespace: str = "zibtek"
    ) -> Dict[str, Any]:
        """Get RAG response from your API with retrieval steps."""
        
        try:
            payload = {
                "question": question,
                "namespace": namespace
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat",
                json=payload,
                headers=self._get_auth_headers(),
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "answer": data.get("answer", ""),
                    "contexts": self._extract_contexts(data),
                    "citations": data.get("citations", []),
                    "retrieval_method": data.get("retrieval_method", "unknown"),
                    "retrieval_steps": data.get("retrieval_steps", {}),
                    "processing_time_ms": data.get("processing_time_ms", 0),
                    "is_out_of_scope": data.get("is_out_of_scope", False)
                }
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return {
                    "answer": f"API Error: {response.status_code}",
                    "contexts": [],
                    "citations": [],
                    "retrieval_method": "error",
                    "retrieval_steps": {},
                    "processing_time_ms": 0,
                    "is_out_of_scope": True
                }
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return {
                "answer": f"Request failed: {str(e)}",
                "contexts": [],
                "citations": [],
                "retrieval_method": "error",
                "retrieval_steps": {},
                "processing_time_ms": 0,
                "is_out_of_scope": True
            }
    
    def _extract_contexts(self, api_response: Dict[str, Any]) -> List[str]:
        """Extract contexts from API response.
        
        This is a simplified approach. In practice, you might need to
        modify your API to return the actual retrieved document texts.
        """
        # For now, we'll use citations as a proxy for contexts
        # In a real implementation, you'd want your API to return actual document texts
        citations = api_response.get("citations", [])
        
        # Placeholder contexts based on citations
        contexts = []
        for citation in citations:
            # This is a simplified approach - ideally your API would return full context
            contexts.append(f"Retrieved content from: {citation}")
        
        return contexts if contexts else ["No context retrieved"]
    
    async def evaluate_test_set(
        self,
        test_cases: List[Dict[str, Any]], 
        namespace: str = "zibtek",
        include_ground_truth_metrics: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a set of test cases using RAGAS."""
        
        print(f"🧪 Evaluating {len(test_cases)} test cases for namespace: {namespace}")
        
        # Collect evaluation data
        evaluation_data = []
        
        for i, test_case in enumerate(test_cases, 1):
            question = test_case["question"]
            ground_truth = test_case.get("ground_truth", "")
            
            print(f"   [{i}/{len(test_cases)}] {question[:50]}...")
            
            # Get RAG response
            response = await self.get_rag_response(question, namespace)
            
            eval_item = {
                "question": question,
                "answer": response["answer"],
                "contexts": response["contexts"],
                "ground_truth": ground_truth if include_ground_truth_metrics else ""
            }
            
            evaluation_data.append(eval_item)
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        # Convert to RAGAS dataset
        dataset = Dataset.from_pandas(pd.DataFrame(evaluation_data))
        
        # Select metrics based on available data
        metrics = [
            faithfulness,           # Answer faithfulness to context
            answer_relevancy,       # Answer relevance to question
            context_precision,      # Precision of retrieved contexts
            context_recall,         # Recall of retrieved contexts
            context_utilization,    # How well context is utilized
        ]
        
        if include_ground_truth_metrics and any(item["ground_truth"] for item in evaluation_data):
            metrics.extend([
                answer_correctness,     # Correctness vs ground truth
                answer_similarity,      # Similarity to ground truth
                context_entity_recall,  # Entity recall in contexts
            ])
        
        print(f"🔍 Running RAGAS evaluation with {len(metrics)} metrics...")
        
        # Run RAGAS evaluation
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            return {
                "namespace": namespace,
                "total_questions": len(test_cases),
                "timestamp": datetime.now().isoformat(),
                "metrics": dict(result),
                "detailed_results": evaluation_data,
                "dataset_info": {
                    "has_ground_truth": include_ground_truth_metrics,
                    "metrics_used": [metric.name for metric in metrics]
                }
            }
            
        except Exception as e:
            print(f"❌ RAGAS evaluation failed: {e}")
            return {
                "namespace": namespace,
                "total_questions": len(test_cases),
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "detailed_results": evaluation_data
            }
    
    async def compare_retrieval_methods(
        self,
        test_cases: List[Dict[str, Any]],
        namespace: str = "zibtek"
    ) -> Dict[str, Any]:
        """Compare different retrieval methods if your API supports it."""
        
        print(f"🔬 Comparing retrieval methods for namespace: {namespace}")
        
        # Note: This assumes your API has different endpoints or parameters
        # for different retrieval methods. Adjust based on your implementation.
        
        methods = ["vector_only", "hybrid", "hybrid_reranked"]
        results = {}
        
        for method in methods:
            print(f"   Testing: {method}")
            
            # For this example, we'll use the same endpoint
            # In practice, you might have different endpoints or parameters
            result = await self.evaluate_test_set(
                test_cases, 
                namespace, 
                include_ground_truth_metrics=False
            )
            
            results[method] = result
        
        return {
            "namespace": namespace,
            "comparison_timestamp": datetime.now().isoformat(),
            "methods_compared": methods,
            "results": results
        }
    
    def generate_report(
        self, 
        results: Dict[str, Any], 
        output_path: str = "evaluation/ragas_report.html"
    ):
        """Generate comprehensive HTML report."""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGAS Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 40px; 
            color: #333; 
        }}
        .metric-card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            margin: 15px 0; 
            border-radius: 8px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }}
        .metric-name {{ font-size: 18px; font-weight: 600; }}
        .metric-score {{ 
            font-size: 24px; 
            font-weight: bold; 
            background: rgba(255,255,255,0.2); 
            padding: 5px 15px; 
            border-radius: 25px; 
        }}
        .namespace-section {{ 
            border-left: 4px solid #667eea; 
            padding-left: 20px; 
            margin: 30px 0; 
        }}
        .details-table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
        }}
        .details-table th, .details-table td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }}
        .details-table th {{ 
            background-color: #667eea; 
            color: white; 
        }}
        .details-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .summary-stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }}
        .stat-box {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center; 
        }}
        .stat-value {{ 
            font-size: 28px; 
            font-weight: bold; 
            color: #667eea; 
        }}
        .stat-label {{ 
            color: #666; 
            margin-top: 5px; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 RAGAS Evaluation Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        
        # Handle both single namespace and comparison results
        if "results" in results:  # Comparison results
            for method, result in results["results"].items():
                html_content += self._generate_namespace_section(method, result)
        else:  # Single namespace results
            html_content += self._generate_namespace_section(
                results.get("namespace", "Unknown"), results
            )
        
        html_content += """
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Report generated: {output_path}")
    
    def _generate_namespace_section(self, name: str, result: Dict[str, Any]) -> str:
        """Generate HTML section for a namespace/method result."""
        
        if "error" in result:
            return f"""
            <div class="namespace-section">
                <h2>❌ {name}</h2>
                <p style="color: red;">Error: {result['error']}</p>
            </div>
            """
        
        metrics = result.get("metrics", {})
        total_questions = result.get("total_questions", 0)
        
        section_html = f"""
        <div class="namespace-section">
            <h2>📊 {name}</h2>
            
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-value">{total_questions}</div>
                    <div class="stat-label">Test Questions</div>
                </div>
        """
        
        # Add metric cards
        for metric_name, score in metrics.items():
            if isinstance(score, (int, float)) and not metric_name.startswith('_'):
                section_html += f"""
                <div class="metric-card">
                    <span class="metric-name">{metric_name.replace('_', ' ').title()}</span>
                    <span class="metric-score">{score:.3f}</span>
                </div>
                """
        
        section_html += """
            </div>
        </div>
        """
        
        return section_html
    
    def save_results(self, results: Dict[str, Any], output_path: str = "evaluation/results.json"):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved: {output_path}")


# Example usage
if __name__ == "__main__":
    async def example_evaluation():
        """Example evaluation run."""
        
        # Sample test cases
        test_cases = [
            {
                "question": "What services does Zibtek offer?",
                "ground_truth": "Zibtek offers custom software development, web development, mobile app development, AI solutions, and consulting services."
            },
            {
                "question": "What technologies does Zibtek use?",
                "ground_truth": "Zibtek uses modern technologies including React, Angular, Python, Node.js, AWS, and various AI/ML frameworks."
            },
            {
                "question": "How can I contact Zibtek for a project?",
                "ground_truth": "You can contact Zibtek through their website contact form, email, or phone to discuss your project requirements."
            },
            {
                "question": "What is the weather like today?",
                "ground_truth": "This should be rejected as out-of-scope for a software development company chatbot."
            }
        ]
        
        # Initialize evaluator
        evaluator = RAGASEvaluator(
            api_base_url="http://localhost:8000",  # or your deployed URL
            auth_token=None  # Set your JWT token if needed
        )
        
        # Run evaluation
        results = await evaluator.evaluate_test_set(test_cases, namespace="zibtek")
        
        # Generate report
        evaluator.generate_report(results, "evaluation/ragas_report.html")
        evaluator.save_results(results, "evaluation/results.json")
        
        print("🎉 Evaluation complete!")
    
    # Run example
    asyncio.run(example_evaluation())