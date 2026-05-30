## 🏋️ Training Scripts (For Reference)

For complete transparency and to support reproducibility, we have included the original training scripts used to develop the RELIC and ProtWord models in the `scripts/training/` directory. 

**⚠️ Important Disclaimer Before Running:**
Please note that these scripts are provided **"as-is"** as raw, standalone Python files. They are intended primarily for reference and educational purposes rather than out-of-the-box execution. If you intend to train your own models from scratch or fine-tune them, you must make the following adjustments:

* **Hardware Configuration:** The scripts were written for our specific multi-GPU distributed training environment (PyTorch DDP). You will need to manually adjust the environment variables, CUDA device allocations, batch sizes, and learning rates to match your specific cluster or local hardware limits.
* **Data Formatting:** The training loops expect input data to be heavily pre-processed and formatted in specific ways (e.g., specific HDF5 structures, tokenized `.pt` files). You must write your own data loaders or pre-process your datasets to exactly match the expected input pipelines before the scripts will run successfully.
* **No CLI Wrapper:** Unlike our inference scripts, these training files do not have a unified Command Line Interface (CLI) and contain hardcoded paths that you must modify.

We encourage researchers to read through the training logic (especially the hierarchical bottleneck and VQ-VAE codebook mixing schedules) and adapt the core PyTorch modules to their own training pipelines.
