# Effective Model Pruning: Measuring the Redundancy of Model Components

Official code for the ICML 2026 spotlight paper *Effective Model Pruning: Measuring the Redundancy of Model Components* by Yixuan Wang, Dan P. Guralnik, Saiedeh Akbari, and Warren E. Dixon.

[Paper on ArXiv](https://arxiv.org/abs/2509.25606) · [BibTeX](#citation) · [Reproducing the tables and figures](#reproducing-the-paper)

## What this repository delivers

Effective Model Pruning (EMP) is a universal threshold selection rule. Given any score vector $s \in \mathbb{R}^N$ produced by any pruning criterion, EMP computes the normalized probability vector $\omega \in \Delta^{N - 1}$ on the standard simplex via

$$\omega_i = \frac{|s_i|}{\sum_{j=1}^{N} |s_j|}, \qquad i = 1, \ldots, N,$$

and returns the effective sample size

$$N_\text{eff}(\omega)= \left( \sum_{i=1}^{N} \omega_i^2 \right)^{-1},$$

namely the inverse Simpson index of $\omega$, also known in the sequential Monte Carlo literature as the effective number of particles. The $N - \nu$ lowest scoring components are pruned, where $\nu := \lfloor \beta\, N_\text{eff} \rfloor$ is the integer retention count and $\beta \in (0, +\infty)$ is the deployment knob whose default value is $\beta = 1$. Theorem 4.2 of the paper establishes a universal tight lower bound on the preserved mass fraction

$$s_\text{eff} = \sum_{i \in \pi} \omega_i \geq \frac{\nu}{N} + \frac{N - \nu}{N}\sqrt{\frac{N - \nu - 1}{(\nu + 1)(N - 1)}},$$

where $\pi$ is the set of indices of the top $\nu$ entries of $\omega$, equivalently written

$$1 - s_\text{eff} \leq \frac{N - \nu}{N}\left(1 - \sqrt{\frac{N - \nu - 1}{(\nu + 1)(N - 1)}}\right).$$

Under the further restriction that the score vector coincides with the parameter vector (the magnitude pruning setting $s = \theta^\ast$), Equation 5 of the paper combines this mass bound with Lemma 3.1 of Zhang et al. (2023) to produce an asymptotic upper bound on the post pruning loss change in the regime $N \to \infty$ with $\rho := \nu / N$ fixed in $(0, 1)$. This loss bound is asymptotic in $N$, not pointwise, and it applies specifically to magnitude pruning; the mass bound on $s_\text{eff}$ above applies to every scoring criterion.

The code reproduces every numerical claim in the paper across six architecture families (fully connected networks, convolutional networks, residual networks, Kolmogorov Arnold networks, GPT 2 attention heads, LLaMA and LLaMA 2 weight matrices), four scoring criteria (weight magnitude, Taylor loss change, gradient saliency, Wanda), three pruning granularities (unstructured weights, attention heads, image pixels and patches), and two settings beyond the submitted manuscript (pre softmax attention edge pruning, and attention sink suppression during training).

## Quick start: the implementation of EMP in PyTorch

The entire method is a few lines of PyTorch. The function below reproduces `EMP_global_magnitude` from `experiments.ipynb` and returns the global sparsity (the fraction of weights pruned) achieved when retaining the top $\lfloor \beta\, N_\text{eff} \rfloor$ weights by magnitude across all convolutional and linear layers of the model.

```python
import torch
import torch.nn as nn

def emp_global_magnitude(model: nn.Module, beta: float = 1.0) -> float:
    """Apply EMP pruning by weight magnitude in place; return global sparsity."""
    params = [m.weight.data.view(-1) for m in model.modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))]
    s = torch.cat(params)
    omega = torch.abs(s) / torch.sum(torch.abs(s))
    n_eff = 1.0 / torch.sum(omega ** 2)
    nu = int(torch.clamp(torch.floor(beta * n_eff), 1, len(s) - 1).item())
    threshold = torch.sort(torch.abs(s), descending=True).values[nu]
    total, kept = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            mask = (torch.abs(m.weight.data) >= threshold)
            m.weight.data.mul_(mask)
            kept += mask.sum().item()
            total += mask.numel()
    return 1.0 - kept / total
```

For criteria where the sign of $s_i$ carries semantic meaning (positive indicating retention, negative indicating removal), apply the monotone shift $s_i \mapsto s_i - \min_j s_j$ before computing $\omega$ so that the full importance ordering is preserved without losing the contribution of negative scores. For criteria where only the magnitude of $s_i$ indicates importance (weight magnitude $|w|$, Taylor importance $|w \cdot \nabla_w \mathcal{L}|$, activation norms, Wanda scores $|w| \cdot \|x\|_2$), the absolute value normalization is the natural choice and is used in every experiment of this repository.

## Installation

The repository targets PyTorch 2.x with CUDA support. The minimal environment for the FC, CNN, KAN, image, and GPT 2 experiments is

```bash
conda create -n emp python=3.10
conda activate emp
pip install -r requirements.txt
```

The LLaMA and LLaMA 2 experiments additionally require a Hugging Face account with access to the `huggyllama/llama-*` and `meta-llama/Llama-2-*` model checkpoints. Place your token in the environment variable `HF_TOKEN` before launching any sbatch job:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
sbatch slurm/llama-7b.sbatch
```

The sbatch files target a Slurm cluster with the `hpg-b200` partition and request $4$ B200 GPUs for the 65B and 70B runs. For smaller models adjust `--gres=gpu:b200:N` and `--mem` to your cluster. For systems without Slurm the underlying entry point is

```bash
cd slurm
python LLama_wanda_magnitude.py \
  --model_id huggyllama/llama-7b \
  --cache_dir ./llm_weights \
  --calib_seq_len 512 \
  --calib_samples 128 \
  --block_size_eval 2048 \
  --beta 1.0 \
  --eval_zeroshot \
  --zs_max_samples 0 \
  --zs_batch_size 8 \
  --zs_max_length 256
```

The seven zero shot tasks evaluated are BoolQ, RTE, HellaSwag, WinoGrande, ARC easy, ARC challenge, and OpenBookQA, matching Table 6 of the paper.

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── train_models.ipynb                    # Section 5.1 dense baselines
├── experiments.ipynb                     # Section 5.1 EMP pruning and β sweep
├── EMP_KAN.ipynb                         # Section 5.2 KAN node pruning
├── EMP_transformer_attention.ipynb       # Section 5.3 heads + Impact Statement edges
├── EMP_image.ipynb                       # Section 5.5 image feature pruning
├── figures/
│   ├── city.jpg                          # source image for Figure 4
│   └── result_city.png                   # reproduction of Figure 4
└── slurm/
    ├── LLama_wanda_magnitude.py          # Section 5.4 LLaMA driver
    ├── llama-{7b,13b,30b,65b}.sbatch     # LLaMA family launchers
    ├── llama2-{7b,13b,70b}.sbatch        # LLaMA 2 family launchers
    └── logs/                             # committed run logs reproducing Tables 3 and 6
```

## Conventions

Throughout the code and the paper, $N$ denotes the total number of model components (weights, attention heads, or image pixels and patches), $\omega \in \Delta^{N - 1}$ is the normalized importance vector on the probability simplex, $N_\text{eff}$ is the effective sample size defined above, $\nu := \lfloor \beta\, N_\text{eff} \rfloor$ is the integer retention count produced by the algorithm with default $\beta = 1$, $s_\text{eff} := \sum_{i \in \pi} \omega_i$ is the preserved mass fraction where $\pi$ contains the indices of the top $\nu$ entries of $\omega$, and the sparsity reported in tables and figure captions is the fraction of components pruned, namely $1 - \nu / N$. The density reported as a complementary quantity in `EMP_image.ipynb` is $\nu / N$. The discretization rule is the floor, applied consistently across all experiments.

## Citation

If you use EMP or any part of this codebase, please cite

```bibtex
@inproceedings{wang2026effective,
  title     = {Effective Model Pruning: Measuring the Redundancy of Model Components},
  author    = {Wang, Yixuan and Guralnik, Dan P. and Akbari, Saiedeh and Dixon, Warren E.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {ICML 2026},
  year      = {2026},
  note      = {Spotlight}
}
```

## License

Released under the MIT License. See `LICENSE`. The paper itself is released under CC BY 4.0 via OpenReview.

## Acknowledgements

The KAN implementation is adapted from `efficient-kan` ([Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)). The Wanda baseline follows the original implementation of Sun et al. (2023). The LLaMA and LLaMA 2 checkpoints are gated and require accepting the upstream licenses on the Hugging Face Hub.
