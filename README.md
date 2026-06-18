# TransformerCompression

TransformerCompression is an IF2211 Strategi Algoritma paper project that studies mixed-precision Transformer compression as a Dynamic Programming problem. The project evaluates fake quantization on BERT-base, estimates per-layer sensitivity, and uses a multiple-choice knapsack formulation to choose the best bit-width allocation under a model-size budget.

## Paper

The final IEEE-style paper is available at:

- [`docs/main.pdf`](docs/main.pdf)
- [`docs/main.tex`](docs/main.tex)

Title:

**Efficient Transformer Compression via Dynamic Programming-Based Mixed-Precision Bit-Width Allocation**

## Key Result

The selected BERT-base configuration is:

```text
[4, 4, 8, 8, 4, 8, 8, 8, 8, 4, 4, 4]
```

Under budget 72, this configuration achieves:

- Estimated compression ratio: **5.33x** against 32-bit storage
- Estimated accuracy loss: **0.044** from an 89.4% baseline
- Estimated post-allocation accuracy: **85.1%**

## Method Summary

Each BERT-base Transformer encoder layer is assigned one bit-width from `{2, 4, 8}`. For each layer-bit option, the experiment records:

- relative storage cost,
- fake-quantized validation accuracy,
- estimated accuracy loss.

The allocation problem is then modeled as a multiple-choice knapsack problem:

- one option must be selected for every layer,
- total storage cost must stay within a budget,
- total estimated accuracy loss is minimized.

Dynamic Programming is used to recover the globally optimal allocation under the additive loss model. Greedy and uniform bit-width baselines are included for comparison.

## Repository Structure

```text
bitweaver/
  allocators.py                         DP, greedy, and uniform allocators

docs/
  main.tex                              IEEE paper source
  main.pdf                              compiled paper
  bert-base/
    sensitivity_table.csv               per-layer fake-quantization results
    comparison_table.csv                DP, greedy, and uniform comparison
  figures/                              paper figures and signature asset

notebooks/
  bitweaver_bert_base_experiment.ipynb  experiment notebook for Kaggle/Colab

tests/
  test_allocators.py                    allocator unit tests
```

## Running the Allocator Tests

```bash
python -m unittest discover -s tests -v
```

## Reproducing the Experiment

The main experiment is designed to be run in a notebook environment such as Kaggle or Google Colab:

```text
notebooks/bitweaver_bert_base_experiment.ipynb
```

The notebook uses BERT-base and SST-2 validation data through the Hugging Face ecosystem. No dataset files are committed to this repository; generated result tables are stored under `docs/bert-base/`.

Typical Python dependencies:

```bash
pip install torch transformers datasets evaluate scikit-learn pandas matplotlib seaborn
```

## Building the Paper

From the `docs/` directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

The compiled output is written to `docs/main.pdf`.

## Notes

The reported compression values estimate weight-storage reduction only. The experiment uses fake quantization, so it does not claim hardware-level low-bit inference speedup or end-to-end latency improvement.
