# Multi-Layer Feature Circuits in Transformers

This repository hosts exploratory notebooks for studying feature circuits inside multi-layer transformer models.
- `Method_1.ipynb` is a guided workflow that walks through setup, prompt suites, logit lens inspection, correlation-based circuit mapping, ablations, and activation patching.
- `method_2.ipynb` is a streamlined pipeline that loads `distilgpt2`, samples 500 WikiText-2 training texts, extracts sparse MLP neurons, clusters them into circuits, and scores neuron selectivity across prompt families while generating summary plots.

## Getting Started

1. Ensure you have Python 3.10+ with JupyterLab or Jupyter Notebook installed (a GPU is optional but helpful).
2. Clone this repository and install the Python dependencies required by your environment (either run the first cell in each notebook or `pip install transformers datasets torch scikit-learn matplotlib seaborn`).
3. Launch Jupyter and open the notebook you want to run. `method_2.ipynb` will download the WikiText-2 train split and `distilgpt2` weights on first execution.

## Outputs (method_2)

- `sparsity_distribution.png`: sparsity histograms for top neurons per layer.
- `circuits_tsne.png`: t-SNE view of clustered sparse neuron circuits.
- `neuron_family_heatmap.png`: activation selectivity across prompt families.

## Next Steps

- Capture dependencies in a requirements.txt once the experimentation stack stabilizes.
- Add additional notebooks for new methods or ablation studies.
- Commit plots or intermediate data products that are helpful for reproducibility.
