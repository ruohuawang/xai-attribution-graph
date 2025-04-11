
# Attribution Graphs for Transformer Models

This repository implements attribution graph analysis for transformer-based language models, providing tools to understand and visualize how information flows through these models.

## Overview

Attribution graphs help us understand how different components of a transformer model interact with each other during inference. This project provides:

1. Tools to compute attribution between model components
2. Cross Layer Transcoder (CLT) implementation for more accurate attribution
3. Visualization utilities for attribution graphs
4. Support for multiple model architectures (Qwen2.5, GPT-2, BERT)

## Features

- **Attribution Analysis**: Compute how information flows between different layers and components of transformer models
- **CLT Implementation**: Train and use causal language transformers for more accurate attribution
- **Multiple Model Support**: Works with Qwen2.5, GPT-2, and BERT architectures
- **Visualization Tools**: Generate network graphs showing information flow

## Installation

```bash
git clone https://github.com/yourusername/attribution_graphs.git
cd attribution_graphs
pip install -r requirements.txt
```

## Usage

### Basic Attribution Analysis

```python
from model_loader import load_model
from attribution_clt import AttributionCLT
from config import CONFIG

# Load model
model, tokenizer = load_model(CONFIG)

# Create attribution analyzer
attribution = AttributionCLT(model, CONFIG)

# Compute attribution for a sample
inputs = tokenizer("Hello, world!", return_tensors="pt").to(CONFIG["device"])
target_idx = 5  # Target token index
attribution_results = attribution.compute_attribution(inputs.input_ids[0], target_idx)

# Visualize results
attribution.visualize_graph(attribution_results, "attribution_graph.png")
```

### Training a Causal Language Transformer (CLT)

```bash
python train_clt.py
```

### Running Full Analysis

```bash
python run_clt_analysis.py
```

## Configuration

The project uses a configuration system defined in `config.py`. Key configuration options include:

- `model_type`: Choose between "qwen2", "gpt2", or "bert"
- `data_path`: Path to the dataset (default: OpenWebText)
- `attribution_samples`: Number of samples for attribution analysis
- `edge_threshold`: Threshold for including edges in the attribution graph

## Data

The project uses the OpenWebText dataset for training and analysis. The data is automatically downloaded and processed when running the scripts.

## Models

Supported models:
- Qwen2.5 (qwen2.5-0.5b-instruct)
- GPT-2 
- BERT (bert-base-uncased)

## References

This implementation is based on the paper:
- [Attribution Graphs for Transformer Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

## License

[MIT License](LICENSE)
