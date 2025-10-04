# Ground the Domain: From Naive RAG to Production Patterns

*RAG System Implementation and Evaluation*

This project implements and evaluates both naive and enhanced RAG (Retrieval-Augmented Generation) systems, demonstrating differences between basic retrieval and production-ready patterns with advanced techniques.

## Executive Summary

This project implements and evaluates a comprehensive RAG system evolution, demonstrating minimal metric improvements and significant performance issues through advanced techniques. The enhanced RAG system incorporates query rewriting using Flan-T5, cross-encoder reranking, and multiprocessing optimization, resulting in improved retrieval precision and answer quality. The system demonstrates production readiness with auto-saving mechanisms, crash recovery capabilities, and scalable multiprocessing architecture.

The enhanced version of RAG was not found to vastly improve quantitative metrics. However, the implementation of multiprocessing was found to greatly improve metrics. The implementation showcases a complete pipeline from data preprocessing through evaluation, with comprehensive metrics including RAGAS faithfulness, answer relevancy, context precision, and context recall measurements.

---

## Directory Structure

```py
├── archive/                                      # Earlier attempts, might be worth looking into
│   ├── data_exploration.ipynb                    # Initial pipeline
│   ├── evaluation_multiprocessing.ipynb          # Initial implementation of multiprocessing
│   ├── evaluation.ipynb                          # Evaluation w/o multiprocessing (w/ results)
│   ├── rag_evaluation_results_multiprocessing.csv
│   ├── rag_evaluation_results.csv
│   ├── rag_generated_answers copy.csv
│   └── rag_generated_answers.csv
├── data/processed/
│   ├── passage_embeddings.npy                    # Generated passage embeddings
│   ├── queries.csv                               # Query dataset
│   ├── query_embeddings.npy                      # Generated query embeddings
│   └── rag_wikipedia_mini.db                     # Milvus vector database
├── phases/
│   ├── phase1_architecture.md
│   ├── phase2_naive_rag_implementation.md
│   ├── phase3_enhanced_rag_system.md
│   ├── phase4_evaluation_report.md
│   └── phase5_technical_report.md
├── results/
│   ├── enhanced_rag_answers.csv                  # Enhanced RAG generated answers
│   ├── naive_rag_answers.csv                     # Naive RAG generated answers
│   └── naive_rag_evaluation.csv                  # Naive RAG evaluation results (template, broken)
├── src/
│   ├── data_setup.ipynb                          # Data preprocessing and database setup
│   ├── enhanced_rag_eval.ipynb                   # Enhanced RAG evaluation notebook
│   ├── enhanced_rag.ipynb                        # Enhanced RAG implementation
│   ├── naive_rag_eval.ipynb                      # Naive RAG evaluation notebook
│   └── naive_rag.ipynb                           # Naive RAG implementation
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas transformers torch sentence-transformers pymilvus ragas datasets matplotlib seaborn gc
```

### 2. Run Complete Evaluation
- Run `data_exploration.ipynb` file.
- Run `naive_rag.ipynb` file.
- Run `enhanced_rag.ipynb` file.
- Run `naive_evaluation.ipynb` file.
- Run `enhanced_evaluation.ipynb` file.

## Features

### Naive RAG System
- Basic retrieval using cosine similarity
- Simple context concatenation
- Standard generation with Flan-T5

### Enhanced RAG System
- Retrieval reranking
- Query rewriting

### Evaluation Metrics
- **Basic Metrics**: F1 Score, Exact Match (EM)
- **RAGAS Metrics**: Faithfulness, Answer Relevancy, Context Recall, Context Precision

## Evaluation Results

The comprehensive evaluation demonstrates the performance characteristics of both naive and enhanced RAG implementations across the 918-query RAG Mini Wikipedia dataset.

### Naive RAG (with Multiprocessing)
This version represents a baseline approach accelerated by processing queries in parallel. It shows strong performance with respectable accuracy.

**Performance Metrics:**
- **Exact Match (EM)**: 43.14%
- **F1 Score**: 52.92%
- **Average Time per Query**: 2.17 seconds

**Key Feature**: The use of multiprocessing delivered a significant ~4x speed improvement compared to sequential processing, making it highly efficient for production deployment.

### Enhanced RAG (with Advanced Features)
This version adds more sophisticated steps—Query Rewriting and Reranking—to improve the context given to the language model, aiming for higher quality answers.

**Performance Metrics:**
- **Exact Match (EM)**: 42.59%
- **F1 Score**: 54.12%
- **Average Time per Query**: 7.50 seconds

**Analysis**: While the enhanced RAG system achieved higher F1 scores (1.2% improvement), the significant increase in processing time (3.5x slower) raises questions about the cost-benefit ratio. The query rewriting component, while theoretically beneficial, may have added unnecessary complexity for straightforward factual questions in this dataset. Obtaining 10 context passages may have also added unnecessary time.

### Performance Comparison Summary

| System | EM Score | F1 Score | Time/Query |
|--------|----------|----------|------------|
| **Naive RAG** | 34.86% | 43.13% | 5.24s |
| **Naive RAG w/ Multiprocessing** | 43.14% | 52.92% | 2.17s |
| **Enhanced RAG** | 42.59% | 54.12% | 7.50s |

**Key Insights:**
- **Multiprocessing** provides the most significant performance improvement
- **Naive RAG** demonstrates better efficiency compared to the current implementation of enhanced RAG
- While the **Enhanced RAG** configuration produced the highest F1 score, the minor gain came at a major performance cost.
- **Query rewriting** may be unnecessary for well-formed questions in this domain. The dataset contains well-formed questions that do not seem to benefit from a rewriting step. This feature adds significant computational overhead without a clear corresponding improvement in the quality of the retrieved context or the final answer.


Future efforts should focus on this configuration as the primary production candidate. The advanced features in the "Enhanced RAG" pipeline are not cost-effective in their current form and should be either heavily optimized or removed to maintain an efficient user experience. However, it is worth noting that we are unsure of how it would perform on qualitative metrics.

### Advanced RAGAs Metrics
The system generates detailed evaluation reports including:

1. **Basic QA Metrics**
   - F1 Score: Word-level overlap between prediction and ground truth
   - EM Score: Exact word-level match

2. **Advanced RAGAs Metrics**
   - Faithfulness: How well answers are grounded in context
   - Answer Relevancy: Relevance of answers to questions
   - Context Recall: How well context covers ground truth
   - Context Precision: Precision of retrieved context

## Notes

- All systems were executed on CPU
- Results are automatically saved to the `results/` directory