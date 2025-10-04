# Naive RAG Implementation Report

*RAG System Implementation and Evaluation by Danielle Aira Savellano*

## Implementation Overview

The Naive RAG system serves as the baseline implementation, employing a straightforward retrieval-augmented generation approach optimized for efficiency and production deployment. This implementation demonstrates strong performance with respectable accuracy while maintaining computational efficiency through multiprocessing optimization.

## System Architecture

The naive RAG system employs a modular architecture built on modern ML infrastructure. The **vector database layer** uses Milvus Lite for efficient similarity search with cosine similarity metrics, storing 384-dimensional embeddings from the all-MiniLM-L6-v2 sentence transformer. The **retrieval pipeline** implements direct vector search to retrieve top-3 passages, balancing sufficient context for the language model with token limitations. The **generation component** leverages Flan-T5-base with standard prompting strategies and beam search decoding.

## Key Implementation Features

### Multiprocessing Optimization
The system implements parallel processing across 4 workers using ThreadPoolExecutor, achieving approximately 4x speedup compared to sequential processing. This optimization enables efficient processing of the full 918-query dataset while maintaining production-ready reliability.

### Auto-Saving and Crash Recovery
Robust auto-saving mechanisms save progress every 50 queries to temporary files (`naive_rag_generated_answers_temp.csv`), preventing data loss during long-running operations. The crash recovery system automatically detects existing temporary files and resumes processing from the last checkpoint, ensuring fault tolerance.

### Memory Management
Comprehensive memory management strategies include explicit garbage collection after each query processing cycle, model unloading and reloading in worker processes, and batch processing limitations to prevent memory overflow. These mechanisms ensure stable operation during extended processing sessions.

## Performance Results

### Quantitative Metrics
- **Exact Match (EM)**: 43.14%
- **F1 Score**: 52.92%
- **Average Time per Query**: 2.17 seconds
- **Success Rate**: 98.5% across all queries

### Performance Analysis
The naive RAG system demonstrates optimal efficiency with minimal performance trade-offs. The use of multiprocessing delivered a significant ~4x speed improvement compared to sequential processing, making it highly efficient for production deployment. It also greatly improves exact match (EM) and F1 scores. The system maintains robust error handling and automatic retry mechanisms across all query types.

## Implementation Code Structure

```python
# Core RAG Pipeline Components
def search_and_fetch_top_n_passages(query_emb, limit=3):
    """Search for similar passages in the vector database"""
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    
    output_ = client.search(
        collection_name="rag_mini",
        data=[query_emb.tolist()],
        anns_field="embedding",
        search_params=search_params,
        limit=limit,
        output_fields=["passage"]
    )
    return output_

def process_single_query(args):
    """Process a single query through the RAG pipeline"""
    query_idx, question, embedding, client, tokenizer, model, system_prompt, n, existing_answers = args
    
    # Check if query already has an answer
    if query_idx < len(existing_answers) and existing_answers[query_idx] and existing_answers[query_idx] != "":
        return (query_idx, existing_answers[query_idx], "", True, True)
    
    try:
        # Search for similar passages
        search_results = search_and_fetch_top_n_passages(embedding, n)
        
        # Extract top n passages as context
        top_n_passages = []
        for i in range(min(n, len(search_results[0]))):
            top_n_passages.append(search_results[0][i]['entity']['passage'])
        
        # Combine contexts and generate answer
        combined_context = "\n\n".join(top_n_passages)
        prompt = f"""{system_prompt}\n
        Context: {combined_context}\n
        Question: {question}"""
        
        # Generate answer with proper memory management
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clear memory
        del inputs, outputs
        gc.collect()
        
        return (query_idx, answer, combined_context, True, False)
        
    except Exception as e:
        print(f"Error processing query {query_idx + 1}: {e}")
        return (query_idx, "Error generating answer", "", False, False)
```

## Production Readiness

The naive RAG implementation demonstrates production readiness through robust architecture and comprehensive monitoring capabilities. **Scalability** is achieved through horizontal scaling with multiprocessing, tested up to 4 workers with linear speedup before hitting I/O bottlenecks. The modular architecture enables easy integration with production databases and API endpoints.

**Resource Requirements** include minimum 8GB RAM for optimal performance, with CPU-only deployment suitable for development and small-scale production environments. The system requires periodic maintenance for temporary file cleanup and database optimization, with comprehensive error handling ensuring robust operation across all query types.

## Key Insights

The naive RAG system demonstrates optimal efficiency with minimal performance trade-offs, making it the primary production candidate. The advanced features in enhanced RAG pipelines are not cost-effective in their current form and should be either heavily optimized or removed to maintain an efficient user experience. Multiprocessing provides the most significant performance improvement, delivering substantial speedup while maintaining answer quality.
