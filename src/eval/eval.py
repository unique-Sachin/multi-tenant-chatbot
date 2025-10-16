"""Evaluation framework for Zibtek chatbot scope detection and groundedness.

This module provides:
1. CSV evaluation dataset loading
2. Automated API testing against /chat endpoint
3. Scope accuracy metrics (in-scope vs out-of-scope)
4. Grounded hit@1 metrics (expected content found)
5. Performance metrics (latency, cost)
6. Results output to JSON
"""

import csv
import json
import time
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests


@dataclass
class EvalQuestion:
    """Single evaluation question."""
    question: str
    expected_contains: str
    in_scope: bool


@dataclass
class EvalResult:
    """Result of evaluating one question."""
    question: str
    expected_contains: str
    in_scope_expected: bool
    
    # Response data
    answer: str
    is_out_of_scope_actual: bool
    citations: List[str]
    processing_time_ms: int
    retrieval_method: str
    
    # Request performance
    request_latency_ms: float
    
    # Computed metrics
    scope_correct: bool
    grounded_hit: bool


class ZibtekEvaluator:
    """Evaluation system for Zibtek chatbot."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.results: List[EvalResult] = []
    
    def load_eval_dataset(self, csv_path: str) -> List[EvalQuestion]:
        """Load evaluation questions from CSV file."""
        questions = []
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                questions.append(EvalQuestion(
                    question=row['question'].strip(),
                    expected_contains=row['expected_contains'].strip(),
                    in_scope=bool(int(row['in_scope']))
                ))
        
        print(f"📊 Loaded {len(questions)} evaluation questions")
        return questions
    
    def call_chat_api(self, question: str) -> Tuple[Dict[str, Any], float]:
        """Call the /chat API and return response data and latency."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/chat",
                json={"question": question},
                timeout=60
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                return response.json(), latency_ms
            else:
                print(f"❌ API error {response.status_code}: {response.text[:100]}")
                return None, latency_ms
                
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            print(f"❌ Request failed: {e}")
            return None, latency_ms
    
    def evaluate_question(self, eval_q: EvalQuestion) -> EvalResult:
        """Evaluate a single question."""
        print(f"🔍 Evaluating: {eval_q.question[:50]}...")
        
        # Call API
        response_data, latency = self.call_chat_api(eval_q.question)
        
        if response_data is None:
            # Handle API failure
            return EvalResult(
                question=eval_q.question,
                expected_contains=eval_q.expected_contains,
                in_scope_expected=eval_q.in_scope,
                answer="API_FAILED",
                is_out_of_scope_actual=True,
                citations=[],
                processing_time_ms=0,
                retrieval_method="failed",
                request_latency_ms=latency,
                scope_correct=False,
                grounded_hit=False
            )
        
        # Extract response fields
        answer = response_data.get('answer', '')
        is_out_of_scope = response_data.get('is_out_of_scope', False)
        citations = response_data.get('citations', [])
        processing_time = response_data.get('processing_time_ms', 0)
        retrieval_method = response_data.get('retrieval_method', 'unknown')
        
        # Compute metrics
        scope_correct = (eval_q.in_scope and not is_out_of_scope) or (not eval_q.in_scope and is_out_of_scope)
        
        # Check if expected content is found (case-insensitive)
        grounded_hit = eval_q.expected_contains.lower() in answer.lower()
        
        return EvalResult(
            question=eval_q.question,
            expected_contains=eval_q.expected_contains,
            in_scope_expected=eval_q.in_scope,
            answer=answer,
            is_out_of_scope_actual=is_out_of_scope,
            citations=citations,
            processing_time_ms=processing_time,
            retrieval_method=retrieval_method,
            request_latency_ms=latency,
            scope_correct=scope_correct,
            grounded_hit=grounded_hit
        )
    
    def run_evaluation(self, questions: List[EvalQuestion]) -> None:
        """Run evaluation on all questions."""
        print(f"🚀 Starting evaluation of {len(questions)} questions...")
        print("=" * 60)
        
        self.results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]", end=" ")
            result = self.evaluate_question(question)
            self.results.append(result)
            
            # Brief result preview
            scope_status = "✅" if result.scope_correct else "❌"
            ground_status = "✅" if result.grounded_hit else "❌"
            print(f"{scope_status} Scope | {ground_status} Ground | {result.request_latency_ms:.0f}ms")
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.5)
        
        print(f"\n✅ Evaluation completed!")
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics from results."""
        if not self.results:
            return {}
        
        # Scope accuracy
        scope_correct_count = sum(1 for r in self.results if r.scope_correct)
        scope_accuracy = scope_correct_count / len(self.results)
        
        # Grounded hit@1 (only for in-scope questions)
        in_scope_results = [r for r in self.results if r.in_scope_expected]
        if in_scope_results:
            grounded_hits = sum(1 for r in in_scope_results if r.grounded_hit)
            grounded_hit_rate = grounded_hits / len(in_scope_results)
        else:
            grounded_hit_rate = 0.0
        
        # Latency metrics
        latencies = [r.request_latency_ms for r in self.results if r.answer != "API_FAILED"]
        avg_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0
        
        # Processing time metrics
        processing_times = [r.processing_time_ms for r in self.results if r.processing_time_ms > 0]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0
        
        # Cost estimation (rough)
        total_processing_time_sec = sum(processing_times) / 1000 if processing_times else 0
        estimated_cost = total_processing_time_sec * 0.00001  # Very rough estimate
        
        # Breakdown by scope
        in_scope_count = len(in_scope_results)
        out_of_scope_results = [r for r in self.results if not r.in_scope_expected]
        out_of_scope_count = len(out_of_scope_results)
        
        # Scope-specific accuracies
        in_scope_correct = sum(1 for r in in_scope_results if r.scope_correct)
        out_of_scope_correct = sum(1 for r in out_of_scope_results if r.scope_correct)
        
        in_scope_accuracy = in_scope_correct / in_scope_count if in_scope_count > 0 else 0
        out_of_scope_accuracy = out_of_scope_correct / out_of_scope_count if out_of_scope_count > 0 else 0
        
        return {
            "evaluation_summary": {
                "total_questions": len(self.results),
                "in_scope_questions": in_scope_count,
                "out_of_scope_questions": out_of_scope_count,
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "scope_metrics": {
                "overall_scope_accuracy": round(scope_accuracy, 3),
                "in_scope_accuracy": round(in_scope_accuracy, 3),
                "out_of_scope_accuracy": round(out_of_scope_accuracy, 3),
                "scope_correct_count": scope_correct_count,
                "scope_total_count": len(self.results)
            },
            "groundedness_metrics": {
                "grounded_hit_at_1": round(grounded_hit_rate, 3),
                "grounded_hits": grounded_hits if in_scope_results else 0,
                "in_scope_total": len(in_scope_results)
            },
            "performance_metrics": {
                "avg_request_latency_ms": round(avg_latency, 1),
                "median_request_latency_ms": round(median_latency, 1),
                "avg_server_processing_ms": round(avg_processing_time, 1),
                "estimated_total_cost_usd": round(estimated_cost, 6)
            },
            "detailed_results": [
                {
                    "question": r.question,
                    "expected_contains": r.expected_contains,
                    "in_scope_expected": r.in_scope_expected,
                    "is_out_of_scope_actual": r.is_out_of_scope_actual,
                    "scope_correct": r.scope_correct,
                    "grounded_hit": r.grounded_hit,
                    "answer_preview": r.answer[:100] + "..." if len(r.answer) > 100 else r.answer,
                    "citations_count": len(r.citations),
                    "request_latency_ms": round(r.request_latency_ms, 1),
                    "processing_time_ms": r.processing_time_ms,
                    "retrieval_method": r.retrieval_method
                }
                for r in self.results
            ]
        }
    
    def print_results_table(self, metrics: Dict[str, Any]) -> None:
        """Print a formatted results table."""
        print("\n" + "="*60)
        print("📊 ZIBTEK CHATBOT EVALUATION RESULTS")
        print("="*60)
        
        # Summary
        summary = metrics["evaluation_summary"]
        print(f"📝 Total Questions: {summary['total_questions']}")
        print(f"✅ In-Scope: {summary['in_scope_questions']}")
        print(f"❌ Out-of-Scope: {summary['out_of_scope_questions']}")
        
        # Scope Metrics
        scope = metrics["scope_metrics"]
        print(f"\n🎯 SCOPE DETECTION ACCURACY")
        print(f"   Overall: {scope['overall_scope_accuracy']:.1%} ({scope['scope_correct_count']}/{scope['scope_total_count']})")
        print(f"   In-Scope: {scope['in_scope_accuracy']:.1%}")
        print(f"   Out-of-Scope: {scope['out_of_scope_accuracy']:.1%}")
        
        # Groundedness Metrics
        ground = metrics["groundedness_metrics"]
        print(f"\n🔍 GROUNDEDNESS (HIT@1)")
        print(f"   Hit Rate: {ground['grounded_hit_at_1']:.1%} ({ground['grounded_hits']}/{ground['in_scope_total']})")
        
        # Performance Metrics
        perf = metrics["performance_metrics"]
        print(f"\n⚡ PERFORMANCE")
        print(f"   Avg Latency: {perf['avg_request_latency_ms']:.0f}ms")
        print(f"   Median Latency: {perf['median_request_latency_ms']:.0f}ms")
        print(f"   Avg Processing: {perf['avg_server_processing_ms']:.0f}ms")
        print(f"   Est. Cost: ${perf['estimated_total_cost_usd']:.6f}")
        
        print("\n" + "="*60)
    
    def save_results(self, output_path: str, metrics: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """Run the evaluation."""
    import os
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server health check failed")
            return
    except:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("   Please make sure the Zibtek chatbot server is running")
        return
    
    print("✅ Server is running and healthy")
    
    # Initialize evaluator
    evaluator = ZibtekEvaluator()
    
    # Load evaluation dataset
    eval_csv_path = "eval.csv"
    if not os.path.exists(eval_csv_path):
        print(f"❌ Evaluation file not found: {eval_csv_path}")
        return
    
    questions = evaluator.load_eval_dataset(eval_csv_path)
    
    # Run evaluation
    evaluator.run_evaluation(questions)
    
    # Compute and display metrics
    metrics = evaluator.compute_metrics()
    evaluator.print_results_table(metrics)
    
    # Save results
    output_path = "eval_results.json"
    evaluator.save_results(output_path, metrics)
    
    print(f"\n🎉 Evaluation complete! Check {output_path} for detailed results.")


if __name__ == "__main__":
    main()