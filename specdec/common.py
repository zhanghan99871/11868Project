"""Shared setup. The project depends on MiniTorch only: no torch, no transformers.

Tokenization uses the `tokenizers` library on the checkpoint's tokenizer.json;
checkpoints are read with safetensors.numpy. `assert_no_torch()` is called at the
end of every entry point to make sure nothing pulled torch in transitively.
"""
import json
import os
import sys

BRIDGES_HF_HOME = "/ocean/projects/cis260267p/yni3/hf"  # shared HF cache on Bridges-2 (jobs/run.sh)


def _use_bridges_hf_cache():
    """On Bridges-2, default HF_HOME to the shared cache. Elsewhere keep Hugging Face's default
    (~/.cache/huggingface). Must run before huggingface_hub is imported."""
    if "HF_HOME" not in os.environ and os.path.isdir(BRIDGES_HF_HOME):
        os.environ["HF_HOME"] = BRIDGES_HF_HOME


_use_bridges_hf_cache()
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(PROJ, "results", "prompts_wikitext2.json")
MODELS = {"gpt2": "openai-community/gpt2", "gpt2-medium": "openai-community/gpt2-medium",
          "distilgpt2": "distilbert/distilgpt2"}  # all share the GPT-2 BPE tokenizer

# eps hard-coded in MiniTorch's fused LayerNorm kernel (LN_EPSILON in src/layernorm_kernel.cu).
# It is 1e-5 (GPT-2's value) since the MLA merge; it was 1e-8 in the copied HW4 kernel.
# tests/test_kernel_consistency.py fails if this drifts from the kernel source.
MINITORCH_LN_EPS = 1e-5

# MLA verifier (GPT-2 medium converted to MLA and fine-tuned, see torch_train/), loaded from a
# torch-free export: python torch_train/export_mla_verifier.py <best.pt> <MLA_VERIFIER_DIR>
MLA_VERIFIER = "mla-gpt2-medium"
MLA_VERIFIER_DIR = os.environ.get(
    "MLA_VERIFIER_DIR", os.path.join(PROJ, "runs", "mla-gpt2-medium", "minitorch"))
TARGETS = ("gpt2-medium", MLA_VERIFIER)
# H-Spec drafter for the MLA verifier: bash train.sh, which exports to this directory
HSPEC_DRAFTER_DIR = os.environ.get(
    "HSPEC_DRAFTER_DIR", os.path.join(PROJ, "runs", "hspec-drafter", "minitorch"))


HF_FILES = ("config.json", "model.safetensors", "tokenizer.json")  # all that specdec reads


def model_dir(name):
    """Local snapshot of checkpoint `name`. Uses the HF cache if it holds HF_FILES; otherwise
    downloads just those files, unless HF_HUB_OFFLINE=1 (Bridges-2 compute nodes, jobs/run.sh)."""
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:  # huggingface_hub < 0.23
        from huggingface_hub.utils import LocalEntryNotFoundError
    try:
        path = snapshot_download(MODELS[name], local_files_only=True)
        if all(os.path.isfile(os.path.join(path, f)) for f in HF_FILES):
            return path
    except LocalEntryNotFoundError:
        pass
    return snapshot_download(MODELS[name], allow_patterns=list(HF_FILES))


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(model_dir("gpt2"), "tokenizer.json"))


def backend():
    import minitorch
    from minitorch.cuda_kernel_ops import CudaKernelOps
    assert minitorch.__file__.startswith(PROJ), "picked up a minitorch outside this project: " + minitorch.__file__
    return minitorch.TensorBackend(CudaKernelOps)


def load(name, max_len=1024, resident=True):
    if name == MLA_VERIFIER:
        from .mla_model import MiniTorchMLAGPT2
        return MiniTorchMLAGPT2(MLA_VERIFIER_DIR, backend(), max_len=max_len, resident=resident)
    from .model import MiniTorchGPT2
    return MiniTorchGPT2(model_dir(name), backend(), max_len=max_len, resident=resident)


def load_gpt2_dir(path, max_len=1024, resident=True):
    """A GPT-2 checkpoint saved with save_pretrained (config.json + model.safetensors), e.g. the
    fine-tuned drafters of torch_train/finetune_distilgpt2.py."""
    from .model import MiniTorchGPT2
    if not os.path.isfile(os.path.join(path, "model.safetensors")):
        raise FileNotFoundError(f"no model.safetensors in {path}")
    return MiniTorchGPT2(path, backend(), max_len=max_len, resident=resident)


def load_hspec_drafter(target, drafter_dir=None, resident=True):
    """The H-Spec drafter (project/minitorch_drafter.py) for a loaded MLA verifier `target`."""
    project = os.path.join(PROJ, "project")
    if project not in sys.path:
        sys.path.insert(0, project)
    from minitorch_drafter import MiniTorchHSpecDrafter
    # the verifier's backend, so the drafter can read the verifier's LM head in place
    shared = getattr(getattr(target, "ops", None), "backend", None)
    return MiniTorchHSpecDrafter.from_verifier(drafter_dir or HSPEC_DRAFTER_DIR, target,
                                               backend=shared or backend(), resident=resident)


def prompts(n=None):
    with open(PROMPTS) as f:
        p = json.load(f)["prompts"]
    return p if n is None else p[:n]


def assert_no_torch():
    assert "torch" not in sys.modules, "torch was imported somewhere"
    print("[check] torch not imported", flush=True)
