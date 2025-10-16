# RAGAS Evaluation Suite

Welcome to the RAGAS evaluation framework for your multi-tenant RAG chatbot!

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install evaluation dependencies
pip install -r requirements-dev.txt
```

### 2. Set Environment Variables
```bash
# Required for RAGAS evaluation
export OPENAI_API_KEY="your_openai_key"

# Optional: API endpoint and authentication
export API_BASE_URL="http://localhost:8000"  # or your deployed URL
export AUTH_TOKEN="your_jwt_token"  # if authentication is required
```

### 3. Run Basic Evaluation
```bash
# Evaluate specific namespace
python evaluation/run_ragas.py --namespace zibtek

# Evaluate all namespaces
python evaluation/run_ragas.py --all

# Use custom test cases
python evaluation/run_ragas.py --namespace zibtek --test-file evaluation/test_cases.json
```

## 📊 RAGAS Metrics Explained

### Core Metrics
- **Faithfulness** (0-1): How well the answer sticks to the retrieved context
- **Answer Relevancy** (0-1): How relevant the answer is to the question
- **Context Precision** (0-1): How precise/relevant the retrieved contexts are
- **Context Recall** (0-1): How much relevant information was retrieved
- **Context Utilization** (0-1): How well the context was used in the answer

### Ground Truth Metrics (when available)
- **Answer Correctness** (0-1): Semantic + factual correctness vs ground truth
- **Answer Similarity** (0-1): Semantic similarity to expected answer
- **Context Entity Recall** (0-1): Entity coverage in retrieved contexts

## 📁 File Structure
```
evaluation/
├── ragas_evaluator.py     # Main evaluation framework
├── run_ragas.py          # CLI tool for running evaluations
├── ragas_analysis.py     # Interactive analysis tools
├── test_cases.json       # Sample test cases
├── README.md             # This file
└── results/              # Generated reports and results
```

## 🛠️ Usage Examples

### Command Line Examples

```bash
# Basic evaluation
python evaluation/run_ragas.py --namespace zibtek

# Evaluate against production API
python evaluation/run_ragas.py \
  --namespace zibtek \
  --api-url https://your-app.herokuapp.com \
  --auth-token your_jwt_token

# Compare retrieval methods (if supported by your API)
python evaluation/run_ragas.py --namespace zibtek --compare-methods

# Custom output directory
python evaluation/run_ragas.py --namespace zibtek --output-dir my_results

# Evaluate all namespaces with custom test cases
python evaluation/run_ragas.py \
  --all \
  --test-file my_custom_tests.json \
  --output-dir batch_evaluation
```

### Python API Examples

```python
import asyncio
from evaluation.ragas_evaluator import RAGASEvaluator

# Initialize evaluator
evaluator = RAGASEvaluator(
    api_base_url="http://localhost:8000",
    auth_token="your_jwt_token"  # optional
)

# Define test cases
test_cases = [
    {
        "question": "What services does your company offer?",
        "ground_truth": "We offer software development, AI solutions, and consulting."
    },
    {
        "question": "How can I contact support?",
        "ground_truth": "You can contact support via email or phone."
    }
]

# Run evaluation
async def main():
    results = await evaluator.evaluate_test_set(test_cases, namespace="zibtek")
    
    # Generate reports
    evaluator.save_results(results, "my_evaluation.json")
    evaluator.generate_report(results, "my_report.html")

asyncio.run(main())
```

### Interactive Analysis

```python
from evaluation.ragas_analysis import RAGASAnalysis

# Load and analyze results
analyzer = RAGASAnalysis()
results = analyzer.load_results("evaluation/results.json")

# Create visualizations
analyzer.plot_metrics_comparison(results)
analyzer.plot_radar_chart(results)

# Get detailed analysis
summary = analyzer.generate_summary_report(results)
detailed_df = analyzer.analyze_detailed_results(results)
```

## 📋 Test Cases Format

Create your test cases in JSON format:

```json
{
  "namespace_name": [
    {
      "question": "What services do you offer?",
      "ground_truth": "Expected answer for comparison (optional)"
    },
    {
      "question": "How can I get support?",
      "ground_truth": "Contact information should be provided"
    }
  ]
}
```

### Test Case Best Practices

1. **Include diverse question types**:
   - Factual questions about your domain
   - How-to questions
   - Contact/support questions
   - Out-of-scope questions (should be rejected)

2. **Ground truth guidelines**:
   - Provide expected answers when possible
   - Keep them concise but comprehensive
   - Include expected rejections for out-of-scope questions

3. **Coverage considerations**:
   - Test different document types (if you have mixed content)
   - Test edge cases and corner cases
   - Include both simple and complex questions

## 📊 Understanding Results

### Metrics Interpretation

| Metric | Score Range | Good Score | What It Means |
|--------|-------------|------------|---------------|
| Faithfulness | 0-1 | >0.8 | Answer doesn't hallucinate beyond context |
| Answer Relevancy | 0-1 | >0.8 | Answer directly addresses the question |
| Context Precision | 0-1 | >0.7 | Retrieved docs are relevant to question |
| Context Recall | 0-1 | >0.7 | All relevant info was retrieved |
| Context Utilization | 0-1 | >0.6 | Context was effectively used |

### Common Issues and Solutions

**Low Faithfulness (<0.6)**:
- Answer contains hallucinations
- Solution: Improve prompts, better context filtering

**Low Answer Relevancy (<0.6)**:
- Answers don't address questions well
- Solution: Better question understanding, prompt tuning

**Low Context Precision (<0.5)**:
- Retrieving too many irrelevant documents
- Solution: Improve embedding quality, tune retrieval parameters

**Low Context Recall (<0.5)**:
- Missing relevant information
- Solution: Increase retrieval count, improve chunking strategy

## 🔧 Advanced Configuration

### Environment Variables
```bash
# RAGAS specific
export OPENAI_API_KEY="sk-..."           # Required for RAGAS LLM calls
export OPENAI_API_BASE="https://..."     # Optional: custom OpenAI endpoint

# Your API configuration
export API_BASE_URL="http://localhost:8000"  # Your RAG API endpoint
export AUTH_TOKEN="jwt_token_here"       # Authentication token

# Optional: Custom model configuration
export RAGAS_LLM_MODEL="gpt-4o-mini"    # LLM for evaluation
export RAGAS_EMBEDDING_MODEL="text-embedding-3-small"  # Embeddings
```

### Custom Metrics

You can extend the evaluation with custom metrics:

```python
from ragas.metrics import Metric

class CustomMetric(Metric):
    """Your custom evaluation metric."""
    
    def score(self, question, answer, contexts, ground_truth=None):
        # Implement your scoring logic
        return score

# Add to evaluation
metrics = [faithfulness, answer_relevancy, CustomMetric()]
```

## 🚀 Integration with CI/CD

Add RAGAS evaluation to your deployment pipeline:

```yaml
# .github/workflows/ragas-evaluation.yml
name: RAGAS Quality Check
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      
      - name: Run RAGAS evaluation
        run: |
          python evaluation/run_ragas.py --namespace zibtek --output-dir ci_results
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          API_BASE_URL: ${{ secrets.STAGING_API_URL }}
          AUTH_TOKEN: ${{ secrets.STAGING_AUTH_TOKEN }}
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: ragas-results
          path: ci_results/
```

## 🎯 Best Practices

### 1. Regular Evaluation
- Run evaluations after each new data ingestion
- Set up automated evaluation on staging environment
- Track metrics over time to detect degradation

### 2. Test Case Management
- Version control your test cases
- Review and update test cases regularly
- Include both positive and negative test cases

### 3. Metric Monitoring
- Set thresholds for acceptable performance
- Monitor trends, not just absolute values
- Use multiple metrics for comprehensive evaluation

### 4. Iterative Improvement
- Use results to guide retrieval tuning
- A/B test different configurations
- Focus on the metrics most important to your use case

## 🆘 Troubleshooting

### Common Issues

**"Import ragas could not be resolved"**:
```bash
pip install -r requirements-dev.txt
```

**"API authentication failed"**:
- Check your AUTH_TOKEN environment variable
- Ensure token is valid and not expired
- Verify API_BASE_URL is correct

**"RAGAS evaluation timeout"**:
- Reduce number of test cases for initial testing
- Check OpenAI API limits and quotas
- Increase timeout in evaluator configuration

**"No contexts retrieved"**:
- Verify your API returns retrieval information
- Check namespace exists and has data
- Test individual API calls manually

### Getting Help

1. Check the [RAGAS documentation](https://docs.ragas.io)
2. Review your API responses manually
3. Start with small test sets to debug issues
4. Enable verbose logging for more details

## 📈 Performance Optimization

### Speed Up Evaluation
- Use smaller test sets for development
- Run evaluations in parallel (be mindful of rate limits)
- Cache evaluation results for repeated analysis

### Cost Optimization
- Use `gpt-4o-mini` instead of `gpt-4` for evaluation
- Limit context length in retrieved documents
- Batch similar questions together

### Memory Management
- Process large test sets in chunks
- Clear results between evaluations
- Use streaming for very large datasets

---

Happy evaluating! 🧪✨