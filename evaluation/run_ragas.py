"""
Command-line interface for running RAGAS evaluations.
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from ragas_evaluator import RAGASEvaluator

# Default test cases for different namespaces
DEFAULT_TEST_CASES = {
    "zibtek": [
        {
            "question": "What services does Zibtek offer?",
            "ground_truth": "Zibtek offers custom software development, web development, mobile app development, AI solutions, and consulting services."
        },
        {
            "question": "What technologies does Zibtek specialize in?",
            "ground_truth": "Zibtek specializes in modern technologies including React, Angular, Python, Node.js, AWS, and various AI/ML frameworks."
        },
        {
            "question": "How can I get a quote from Zibtek?",
            "ground_truth": "You can get a quote from Zibtek by contacting them through their website contact form, email, or phone."
        },
        {
            "question": "What are Zibtek's core competencies?",
            "ground_truth": "Zibtek's core competencies include custom software development, AI/ML solutions, cloud architecture, and digital transformation."
        },
        {
            "question": "Does Zibtek provide ongoing support?",
            "ground_truth": "Yes, Zibtek provides ongoing support and maintenance for the software solutions they develop."
        },
        {
            "question": "What is the weather today?",
            "ground_truth": "This question should be rejected as out-of-scope for a software development company."
        },
        {
            "question": "Can you help me with my math homework?",
            "ground_truth": "This question should be rejected as out-of-scope for a software development company."
        }
    ],
    "generic": [
        {
            "question": "What services are offered?",
            "ground_truth": "Various services are offered based on the organization."
        },
        {
            "question": "How can I contact support?",
            "ground_truth": "Contact information should be available through the website or documentation."
        },
        {
            "question": "What are the pricing options?",
            "ground_truth": "Pricing information should be available or can be requested through contact."
        }
    ]
}

async def run_evaluation(
    namespace: str,
    test_file: str = None,
    api_url: str = None,
    auth_token: str = None,
    output_dir: str = "evaluation",
    compare_methods: bool = False
):
    """Run RAGAS evaluation."""
    
    # Load test cases
    if test_file and os.path.exists(test_file):
        print(f"📋 Loading test cases from: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            all_test_cases = json.load(f)
    else:
        print("📋 Using default test cases")
        all_test_cases = DEFAULT_TEST_CASES
    
    # Get test cases for namespace
    if namespace in all_test_cases:
        test_cases = all_test_cases[namespace]
    else:
        print(f"❌ No test cases found for namespace: {namespace}")
        print(f"Available namespaces: {list(all_test_cases.keys())}")
        return
    
    print(f"🧪 Found {len(test_cases)} test cases for namespace: {namespace}")
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(
        api_base_url=api_url or os.getenv("API_BASE_URL", "http://localhost:8000"),
        auth_token=auth_token or os.getenv("AUTH_TOKEN")
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if compare_methods:
        print(f"🔬 Comparing retrieval methods for: {namespace}")
        results = await evaluator.compare_retrieval_methods(test_cases, namespace)
        
        # Save results
        output_file = f"{output_dir}/comparison_{namespace}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        evaluator.save_results(results, f"{output_file}.json")
        evaluator.generate_report(results, f"{output_file}.html")
        
    else:
        print(f"🧪 Evaluating namespace: {namespace}")
        results = await evaluator.evaluate_test_set(test_cases, namespace)
        
        # Save results
        timestamp = results.get("timestamp", "").replace(":", "").replace("-", "")[:15]
        output_file = f"{output_dir}/evaluation_{namespace}_{timestamp}"
        evaluator.save_results(results, f"{output_file}.json")
        evaluator.generate_report(results, f"{output_file}.html")
    
    print("🎉 Evaluation complete!")

async def run_all_namespaces(
    test_file: str = None,
    api_url: str = None,
    auth_token: str = None,
    output_dir: str = "evaluation"
):
    """Run evaluation for all available namespaces."""
    
    # Load test cases
    if test_file and os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            all_test_cases = json.load(f)
    else:
        all_test_cases = DEFAULT_TEST_CASES
    
    print(f"🧪 Running evaluation for all namespaces: {list(all_test_cases.keys())}")
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(
        api_base_url=api_url or os.getenv("API_BASE_URL", "http://localhost:8000"),
        auth_token=auth_token or os.getenv("AUTH_TOKEN")
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for namespace, test_cases in all_test_cases.items():
        print(f"\n📊 Evaluating namespace: {namespace}")
        
        try:
            results = await evaluator.evaluate_test_set(test_cases, namespace)
            all_results[namespace] = results
            
        except Exception as e:
            print(f"❌ Failed to evaluate {namespace}: {e}")
            all_results[namespace] = {"error": str(e)}
    
    # Generate combined report
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/all_namespaces_{timestamp}"
    
    evaluator.save_results(all_results, f"{output_file}.json")
    evaluator.generate_report(all_results, f"{output_file}.html")
    
    print(f"🎉 All evaluations complete! Results saved to: {output_file}")

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Tool for Multi-tenant RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate specific namespace
  python run_ragas.py --namespace zibtek
  
  # Evaluate with custom test file
  python run_ragas.py --namespace zibtek --test-file my_tests.json
  
  # Evaluate against production API
  python run_ragas.py --namespace zibtek --api-url https://your-app.herokuapp.com
  
  # Compare retrieval methods
  python run_ragas.py --namespace zibtek --compare-methods
  
  # Evaluate all namespaces
  python run_ragas.py --all
  
  # With authentication
  python run_ragas.py --namespace zibtek --auth-token your_jwt_token
        """
    )
    
    parser.add_argument(
        "--namespace", 
        type=str, 
        help="Namespace to evaluate (e.g., zibtek, masai-school)"
    )
    
    parser.add_argument(
        "--test-file", 
        type=str, 
        help="JSON file with test cases"
    )
    
    parser.add_argument(
        "--api-url", 
        type=str, 
        default=None,
        help="API base URL (default: http://localhost:8000 or API_BASE_URL env var)"
    )
    
    parser.add_argument(
        "--auth-token", 
        type=str, 
        default=None,
        help="JWT authentication token (or set AUTH_TOKEN env var)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation",
        help="Output directory for results (default: evaluation)"
    )
    
    parser.add_argument(
        "--compare-methods", 
        action="store_true",
        help="Compare different retrieval methods"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Evaluate all available namespaces"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.namespace:
        parser.error("Must specify either --namespace or --all")
    
    # Import datetime here to avoid circular import issues
    from datetime import datetime
    
    # Run evaluation
    if args.all:
        asyncio.run(run_all_namespaces(
            test_file=args.test_file,
            api_url=args.api_url,
            auth_token=args.auth_token,
            output_dir=args.output_dir
        ))
    else:
        asyncio.run(run_evaluation(
            namespace=args.namespace,
            test_file=args.test_file,
            api_url=args.api_url,
            auth_token=args.auth_token,
            output_dir=args.output_dir,
            compare_methods=args.compare_methods
        ))

if __name__ == "__main__":
    main()