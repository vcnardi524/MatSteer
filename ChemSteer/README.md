# Steering Vector Analysis and Cross-Validation

This folder contains scripts and data for analyzing, extracting, and validating steering vectors in molecular datasets using various models (e.g., GPT-2, Llama). The workflow supports clustering, subspace analysis, threshold calculation, and cross-validation for metrics such as SA (Synthetic Accessibility), QED, and LogP.

## Folder Structure

- `analyze_subspace_clustering.py`  
  Analyze clustering structure in the steering vector subspace.
- `analyze_subspace_coherence.py`  
  Analyze incoherence of clusters in the subspace.
- `calculate_thresholds.py`  
  Calculate metric thresholds for dataset partitioning.
- `crossvalidation_unified.py`  
  Unified script for creating and evaluating local and global steering vectors across metrics and models.
- `extract_gpt.py`, `extract_llama.py`  
  Extract latent vectors from GPT or Llama models for molecules.
- `generate_from_steer_new.py`  
  Generate new molecular samples using steering vectors.
- `thresholds_summary.csv`  
  Summary of calculated thresholds for each dataset/metric.

## Usage

1. **Extract steering vectors:**
   - Use `extract_gpt.py` or `extract_llama.py` to generate latent vectors for a dataset from a model.
2. **Calculate thresholds:**
   - Run `calculate_thresholds.py` to compute metric thresholds for partitioning.
3. **Analyze clusters/subspaces:**
   - If you want to do cluster analysis, use `analyze_subspace_clustering.py` and `analyze_subspace_coherence.py`.
4. **Cross-validation:**
   - Use `crossvalidation_unified.py` to create a global steering vector
5. **Generate molecules:**
   - Use `generate_from_steer_new.py` to generate new molecules from steering vectors using the global steer.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, rdkit, torch, transformers

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Notes
- Thresholds for metrics are summarized in `thresholds_summary.csv`.
- Scripts are modular and can be run independently for different stages of the workflow.

---
For more details, see comments in each script or contact <anonymous authors>.
