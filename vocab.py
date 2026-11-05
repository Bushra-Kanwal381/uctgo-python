"""
Shared vocabulary construction — the honest version of the cross-tokenizer
alignment described in the paper (§7.3).

Problem the old code had:
  * it used a 97-word hand-picked list, not a real cross-tokenizer set;
  * it kept only each word's *first* subword id (`a_id[0]`, `b_id[0]`),
    silently corrupting any multi-subword clinical term.

Fix here:
  * V_shared is a word list (a broad clinical + function-word lexicon, or any
    list you pass). For each word we build a **mean-pooled** input embedding in
    *every* model's embedding space, so a word that splits into several subwords
    is represented by the average of those subword embeddings instead of being
    truncated. This removes the truncation bug and generalises to multi-token
    medical terms.
  * The optimised tensor Z has shape (L, |V_shared|); one continuous
    distribution over whole words per position. The same distribution drives
    the retriever, the generator and BioGPT via three aligned tables (A/B/C),
    which is exactly the "single shared continuous soft-embedding space" claim.

For the BioGPT fluency readout we additionally keep each word's first-subword id
(the only place a first-token projection is used, and it is documented as a
surrogate — the *input* embedding is still mean-pooled).
"""
from dataclasses import dataclass
import torch

from lexicon import DEFAULT_LEXICON


@dataclass
class SharedVocab:
    words: list                 # list[str], length |V|
    W_A: torch.Tensor           # (|V|, d_A)  mean-pooled article-encoder embeds
    W_B: torch.Tensor           # (|V|, d_B)  mean-pooled generator embeds
    W_C: torch.Tensor           # (|V|, d_C)  mean-pooled BioGPT embeds
    bio_first_ids: torch.Tensor # (|V|,) first BioGPT subword id per word (fluency readout)
    cls_A: torch.Tensor         # (d_A,) article [CLS] embedding
    sep_A: torch.Tensor         # (d_A,) article [SEP] embedding
    word_index: dict            # word -> row index

    def __len__(self):
        return len(self.words)


def _mean_embed(tok, emb_layer, word, device, dt):
    # Tables are kept in float32 regardless of model dtype: the optimised tensor
    # Z/soft is float32, so the soft @ W matmul runs in float32 and the *result*
    # is cast to the model dtype just before each model forward (see losses.py).
    ids = tok(word, add_special_tokens=False)["input_ids"]
    if len(ids) == 0:
        return None, None
    ids_t = torch.tensor(ids, device=device)
    with torch.no_grad():
        vecs = emb_layer(ids_t).float()    # (n_sub, d)
        mean = vecs.mean(dim=0)            # (d,) float32
    return mean, ids[0]


def build_shared_vocab(pipe, words=None):
    """Build the aligned mean-pooled tables from a word list.

    A word is kept only if it is non-empty under *all three* tokenizers, so the
    three tables stay row-aligned. Reports the final size.
    """
    words = list(dict.fromkeys(words or DEFAULT_LEXICON))  # dedupe, keep order
    device, dt = pipe.device, pipe.dt

    emb_a = pipe.emb_article()
    emb_b = pipe.emb_gen()
    emb_c = pipe.emb_flu()

    kept, rows_a, rows_b, rows_c, bio_first = [], [], [], [], []
    for w in words:
        ma, _ = _mean_embed(pipe.a_tok, emb_a, w, device, dt)
        mb, _ = _mean_embed(pipe.g_tok, emb_b, w, device, dt)
        mc, first_c = _mean_embed(pipe.f_tok, emb_c, w, device, dt)
        if ma is None or mb is None or mc is None:
            continue
        kept.append(w)
        rows_a.append(ma); rows_b.append(mb); rows_c.append(mc)
        bio_first.append(first_c)

    W_A = torch.stack(rows_a)
    W_B = torch.stack(rows_b)
    W_C = torch.stack(rows_c)
    bio_first_ids = torch.tensor(bio_first, device=device)

    # [CLS]/[SEP] embeddings for a faithful article-encoder forward pass
    cls_id = pipe.a_tok.cls_token_id
    sep_id = pipe.a_tok.sep_token_id
    with torch.no_grad():
        cls_A = emb_a(torch.tensor([cls_id], device=device))[0].float()
        sep_A = emb_a(torch.tensor([sep_id], device=device))[0].float()

    print(f"Shared vocabulary: {len(kept)} words "
          f"(d_A={W_A.shape[1]}, d_B={W_B.shape[1]}, d_C={W_C.shape[1]})")
    return SharedVocab(
        words=kept, W_A=W_A, W_B=W_B, W_C=W_C,
        bio_first_ids=bio_first_ids, cls_A=cls_A, sep_A=sep_A,
        word_index={w: i for i, w in enumerate(kept)},
    )
