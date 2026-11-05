"""
Model loading + lightweight memory management.

Loads MedCPT (query + article encoders), the Qwen generator, and BioGPT.
All models are frozen (no parameter is ever updated — only the soft passage
tensor Z is optimised). The transfer generator is loaded lazily and on its own
so it never has to coexist with the attack models on a 15 GB GPU.
"""
import gc
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def _dtype(name):
    return {"float16": torch.float16, "float32": torch.float32,
            "bfloat16": torch.bfloat16}[name]


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


class Pipeline:
    """Holds the frozen white-box models and their tokenizers."""

    def __init__(self, mcfg):
        self.cfg = mcfg
        self.device = torch.device(mcfg.device if torch.cuda.is_available()
                                   else "cpu")
        dt = _dtype(mcfg.dtype) if self.device.type == "cuda" else torch.float32
        self.dt = dt

        print("Loading MedCPT query encoder ...")
        self.q_tok = AutoTokenizer.from_pretrained(mcfg.retriever_query_model)
        self.q_enc = _freeze(AutoModel.from_pretrained(
            mcfg.retriever_query_model, torch_dtype=dt,
            low_cpu_mem_usage=True)).to(self.device)

        print("Loading MedCPT article encoder ...")
        self.a_tok = AutoTokenizer.from_pretrained(mcfg.retriever_article_model)
        self.a_enc = _freeze(AutoModel.from_pretrained(
            mcfg.retriever_article_model, torch_dtype=dt,
            low_cpu_mem_usage=True)).to(self.device)

        print("Loading Qwen generator ...")
        self.g_tok = AutoTokenizer.from_pretrained(mcfg.generator_model)
        if self.g_tok.pad_token is None:
            self.g_tok.pad_token = self.g_tok.eos_token
        self.gen = _freeze(AutoModelForCausalLM.from_pretrained(
            mcfg.generator_model, torch_dtype=dt,
            low_cpu_mem_usage=True)).to(self.device)
        if mcfg.grad_checkpointing:
            self.gen.gradient_checkpointing_enable()
            self.gen.config.use_cache = False

        print("Loading BioGPT fluency model ...")
        self.f_tok = AutoTokenizer.from_pretrained(mcfg.fluency_model)
        self.flu = _freeze(AutoModelForCausalLM.from_pretrained(
            mcfg.fluency_model, torch_dtype=dt,
            low_cpu_mem_usage=True)).to(self.device)

        free()
        print("All white-box models loaded on", self.device)

    # -- embedding tables (frozen input embeddings) ------------------------
    def emb_article(self):
        return self.a_enc.get_input_embeddings()

    def emb_gen(self):
        return self.gen.get_input_embeddings()

    def emb_flu(self):
        return self.flu.get_input_embeddings()


def load_transfer_generator(mcfg):
    """Load the cross-family transfer generator on its own (call after freeing
    the attack models, or in a fresh process). Returns (tokenizer, model)."""
    device = torch.device(mcfg.device if torch.cuda.is_available() else "cpu")
    dt = _dtype(mcfg.dtype) if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(mcfg.transfer_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = _freeze(AutoModelForCausalLM.from_pretrained(
        mcfg.transfer_model, torch_dtype=dt, low_cpu_mem_usage=True,
        trust_remote_code=True)).to(device)
    return tok, model
