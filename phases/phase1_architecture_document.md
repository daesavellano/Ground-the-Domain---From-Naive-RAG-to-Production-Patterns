# Architecture Document

*RAG System Implementation and Evaluation by Danielle Aira Savellano*

## System Overview

This document outlines the architecture of a Retrieval-Augmented Generation (RAG) system designed to dynamically retrieve relevant information from a knowledge base and leverage it to generate informed, context-aware answers with a large language model. The system implements both naive and enhanced RAG approaches, demonstrating the evolution from basic retrieval to production-ready patterns with advanced techniques.

## Knowledge Base and Data Source

The system's knowledge base is built from the **RAG Mini Wikipedia** dataset, available on **Hugging Face**. This dataset provides a structured corpus for training, testing, and validation with 3,200 text passages and 918 corresponding question-answer pairs. A preliminary analysis revealed significant variance in passage length (1 to 2,151 characters, average 390), with complete data coverage and diverse subject areas enabling comprehensive retrieval quality evaluation.

## System Components and Workflow

### Naive RAG Implementation

The naive RAG system serves as the baseline implementation, employing a straightforward retrieval-augmented generation approach. The system begins with the embedding layer, converting both text passages and user queries into dense vector representations using `all-MiniLM-L6-v2` from the Sentence Transformers library, generating 384-dimensional embeddings. This model was chosen for its balance between performance and computational efficiency, ensuring zero operational cost, local execution control, and perfect reproducibility without external dependencies.

Once vectorized, embeddings are stored and indexed in Milvus Lite with a schema comprising ID, passage text, and vector embedding. When a user submits a query, it's encoded into a vector and used for similarity search to retrieve top-3 most relevant passages. Retrieved passages are concatenated and combined with the original question into a formatted prompt, then passed to Google's Flan-T5-base model for answer generation.

### Enhanced RAG Implementation

The enhanced RAG system incorporates advanced techniques to improve retrieval quality and answer generation. The system implements a two-stage retrieval process with query rewriting and cross-encoder reranking. **Query Rewriting** uses Flan-T5-base to expand and refine user queries, transforming simple questions into more searchable forms with additional context. **Two-Stage Retrieval Process** first retrieves top-10 candidates using cosine similarity search, then employs cross-encoder reranking (ms-marco-MiniLM-L-6-v2) to reorder passages by relevance, selecting the final top-3 most relevant passages for context generation.

**Cross-Encoder Reranking** evaluates query-passage pairs to provide relevance scores, effectively filtering out contextually similar but irrelevant content, significantly improving context precision compared to naive retrieval methods. **Multiprocessing Architecture** supports parallel processing across multiple workers using ThreadPoolExecutor, enabling scalable processing of large query sets while maintaining fault tolerance through auto-saving and crash recovery mechanisms.

## Production-Ready Features

The system incorporates several production-ready features to ensure reliability, scalability, and fault tolerance in real-world deployments. **Auto-Saving and Crash Recovery** mechanisms save progress every 50 queries to temporary files, preventing data loss during long-running operations and enabling seamless resumption of interrupted processing sessions. **Multiprocessing and Scalability** support parallel processing across 4 workers, achieving approximately 3.75x speedup compared to sequential processing, with sophisticated memory management and progress tracking.

**Memory Management** implements comprehensive strategies including explicit garbage collection after each query processing cycle, model unloading and reloading in worker processes, and batch processing limitations to prevent memory overflow. **Progress Tracking and Monitoring** provides real-time visibility into processing status using tqdm progress bars, with comprehensive error handling and automatic retry mechanisms ensuring robust operation across all query types.

## Technical Stack and Environment

This project was developed using open-source technologies and libraries: **Core Technologies** include Python 3.11 and Milvus Lite for local, serverless vector indexing and search. **Key Python Libraries** encompass Sentence Transformers for generating dense vector embeddings, Hugging Face Transformers for loading and running the Flan-T5-base generative model, Hugging Face Datasets for managing the RAG Mini Wikipedia dataset, and Hugging Face Evaluate for calculating SQuAD-based metrics.

**Execution Environment** utilizes CPU-only deployment, ensuring broad compatibility and allowing the architecture to run without requiring specialized GPU hardware. Development and experimentation were conducted primarily within Jupyter Notebooks, with the system designed for local execution control and perfect reproducibility without external dependencies.

## Deployment and Scalability Considerations

The RAG system demonstrates production readiness through robust architecture and comprehensive monitoring capabilities. **Scalability Architecture** achieves horizontal scaling through multiprocessing with ThreadPoolExecutor, tested up to 4 workers with linear speedup before hitting I/O bottlenecks, while vertical scaling is implemented through model optimization and caching strategies. The Milvus Lite vector database supports up to 1M documents with sub-second query response times.

**Resource Requirements** include minimum 8GB RAM for optimal performance, with CPU-only deployment suitable for development and small-scale production environments. Performance characteristics show CPU-only processing limiting throughput to ~75 queries/minute, with memory constraints limiting batch processing to 50 queries per worker. **Deployment Recommendations** include Docker containerization for consistent environments, implementation of health checks and monitoring for the vector database, and automated backup procedures for the embedding store.

**Future Enhancements** should focus on GPU acceleration, distributed processing across multiple machines, and dynamic model loading to support real-time updates and improved fault tolerance, providing 3-5x speedup for embedding generation and reranking operations in larger deployments.
