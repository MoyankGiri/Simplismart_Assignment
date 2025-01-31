# Simplismart: Machine Learning Trainee Assignment

## Author
**Moyank Giri**

- **Position:** M.Tech in Data Science and Artificial Intelligence, IIT Bhilai; J1 Research Scholar, Ohio State University.
- **Experience:** 2+ years in Machine Learning, Deep Learning, and Data Science.
- **Skills:** Python, Computer Vision, NLP, Advanced RAG, Split Learning, Federated Learning, Generative AI, LLM, Explainable AI, CNN, LSTM/RNN, Distributed Deep Learning, ETL, Data Modeling and Mining, Docker, Kubernetes

## Contact
For any queries or suggestions, feel free to reach out:
- **Email:** moyankgiri@example.com  
- **LinkedIn:** [Moyank Giri](https://linkedin.com/in/moyankgiri)

## Overview
This project provides a framework for benchmarking and running inference with large language models (LLMs). It focuses on optimizing the model, given the model name as on HuggingFace, and benchmarkes throughput performance. The framework is designed to support multiple models and measure their suitability for high-throughput inference tasks. It also support interactive mode querying.

## Features
- **Model Loading and Optimization:** Automatically optimizes the model for efficient inference using device maps and ensures proper layer placement based on available GPU memory.
- **Throughput Benchmarking:** Measures the throughput of the model to determine if it meets the desired performance requirements.
- **Interactive Inference:** Allows users to interact with the model in real time.
- **Support for Multiple Models:** Easily benchmark and infer with different pre-trained models.
- **Support for LoRA models:** Easily benchmark and infer with LoRA models.

## Requirements
The following Python libraries are required in addition to the libraries existing on Google Colab:

- `bitsandbytes`
- `transformers`
- `torch`
- `accelerate`

To install the required packages, run:
```bash
pip install -U bitsandbytes transformers
```

## How It Works
The framework processes models using the following steps:

1. **Load and Optimize Model:**
   - Loads the model and tokenizer.
   - Ensures the tokenizer has a padding token.
   - Optimizes model loading with efficient memory placement using `infer_auto_device_map`.

2. **Benchmark Model:**
   - Generates dummy prompts to test model throughput and compares throughput against a target benchmark (200 tokens/sec).

3. **Interactive Inference:**
   - Allows users to enter custom prompts and get model-generated responses.

## Script Usage
### Model Benchmarking
The script benchmarks multiple models specified in the `model_names` list. By default, the following models are included:

- `mistralai/Mistral-7B-v0.1`
- `argilla/notus-7b-v1-lora`

You can add or replace models in the `model_names` list as needed.

### Run Interactive Inference
Uncomment the following line in the script to enable interactive inference:
```python
run_inference(model, tokenizer)
```

### Example Execution
The script will do the following:
1. Load and optimize the model.
2. Benchmark throughput.
3. Run interactive inference.

using the command
```cmd
python3 moyankgiri_simplismartassignment.py
```


### Output Example
```
Processing model: mistralai/Mistral-7B-v0.1
Loading and optimizing the model...
Loading checkpoint shards: 100%
 2/2 [00:01<00:00,  1.85it/s]
Loading checkpoint shards: 100%
 2/2 [01:15<00:00, 35.10s/it]
Running benchmarks...
Benchmark Results:
  Total Throughput: 589.97 tokens/sec
  Target Throughput: 200 tokens/sec
Benchmark passed! The model meets the throughput requirement.
Completed benchmarking for: mistralai/Mistral-7B-v0.1
```

## File Structure
- **`moyankgiri_simplismartassignment.py`**: Main script for loading, benchmarking, and running inference with models.
- **`ReadMe.md`**: File with general information and instructions for easier use 
