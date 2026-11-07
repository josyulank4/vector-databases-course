"""Deterministic, stdlib-only helpers shared by every Course 8 demo package.

One small corpus of support-center articles, a hashing-trick embedder whose
every number a student can reproduce by hand, and the ranking metrics the
whole course grades itself with. No randomness without a fixed seed, no
network, no third-party packages.
"""
from __future__ import annotations

import hashlib
import math
import re

DIM = 256

_TOKEN = re.compile(r"[a-z0-9]+")

# Function words carry positions, not meaning; every production tokenizer
# drops or down-weights them. Dropping them here keeps the toy honest:
# similarity should come from content, not from "the".
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "can", "you", "your", "within", "every", "from", "under",
})


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]

# (doc_id, tenant, category, title, body) — a tiny support-center corpus.
# The semantic trap is deliberate: doc 'reset-2fa' answers the query
# "locked out of my account" while sharing almost no words with it.
DOCS = [
    ("refund-policy", "acme", "billing", "Refund policy for subscriptions",
     "You can request a refund within thirty days of purchase from the billing page. Refunds return to the original payment method within five business days."),
    ("refund-fee", "acme", "billing", "Restocking fee on returned hardware",
     "Returned devices carry a fifteen percent restocking fee unless the unit arrived damaged or defective."),
    ("reset-2fa", "acme", "account", "Regain access when your authenticator is lost",
     "If you cannot sign in because your phone with the authenticator app is gone, use a saved recovery code or ask support to verify your identity and reset two factor authentication."),
    ("password-reset", "acme", "account", "Change or reset your password",
     "To reset a forgotten password, open the sign in page, choose forgot password, and follow the emailed link within one hour."),
    ("invoice-copy", "acme", "billing", "Download a copy of your invoice",
     "Invoices and receipts for every charge are available under billing history. Download the pdf for expense reports."),
    ("shipping-delay", "acme", "orders", "Why is my parcel late",
     "Parcels can be delayed by weather, customs, or a failed delivery attempt. Tracking shows the latest scan and the revised delivery date."),
    ("damaged-parcel", "acme", "orders", "Report a damaged delivery",
     "Photograph the packaging and the item, keep all packing material, and file a damage claim within fourteen days to receive a replacement or credit."),
    ("api-keys", "globex", "developer", "Create and rotate API keys",
     "Generate keys in the developer console. Rotate keys every ninety days and revoke a leaked key immediately."),
    ("rate-limits", "globex", "developer", "Rate limits and quotas",
     "The api allows one thousand requests per minute per key. Exceeding the quota returns http 429 with a retry after header."),
    ("data-export", "globex", "account", "Export your workspace data",
     "Workspace owners can export all documents, messages, and files as a zip archive from the admin settings page."),
    ("sso-setup", "globex", "account", "Configure single sign on",
     "Connect your identity provider with saml or oidc to let the team sign in with company credentials."),
    ("webhook-retries", "globex", "developer", "Webhook delivery and retries",
     "Failed webhook deliveries retry with exponential backoff for twenty four hours before the endpoint is disabled."),
]

# Relevance judgments for the eval queries: doc_id -> grade (0..3).
QUERIES = [
    ("locked out of my account",
     {"reset-2fa": 3, "password-reset": 2, "sso-setup": 1}),
    ("money back for a purchase",
     {"refund-policy": 3, "refund-fee": 1, "invoice-copy": 1}),
    ("my package arrived broken",
     {"damaged-parcel": 3, "shipping-delay": 1}),
]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embed(text: str, dim: int = DIM) -> list[float]:
    """Character-trigram hashing embedding — the mechanism of learned
    embeddings, made inspectable.

    Each word contributes its character trigrams; every trigram claims
    its own bucket (bucket = hash of the trigram, so two different
    trigrams never cancel each other's vote). Paraphrases that share
    word *forms* (refund / refunds, purchase / purchasing) land near
    each other, while unrelated vocabulary stays near-orthogonal — the
    toy version of what a learned model does with subword tokens.
    Deterministic: same text, same vector, on every machine, forever.
    """
    vec = [0.0] * dim
    for tok in content_tokens(text):
        padded = f"#{tok}#"
        grams = {padded[i:i + 3] for i in range(len(padded) - 2)}
        for gram in grams:
            h = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(h[:2], "little") % dim
            vec[idx] += 1.0
    return vec


def normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def doc_vectors() -> dict[str, list[float]]:
    """Title counts twice: production pipelines weight titles and headings
    above body text, because titles are the author's own summary."""
    return {d[0]: normalize(embed(d[3] + " " + d[3] + " " + d[4])) for d in DOCS}


def brute_force(query_vec: list[float], vectors: dict[str, list[float]],
                k: int = 5) -> list[tuple[str, float]]:
    scored = [(doc_id, cosine(query_vec, vec)) for doc_id, vec in vectors.items()]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored[:k]


def bm25_scores(query: str) -> dict[str, float]:
    """Textbook BM25 (k1=1.5, b=0.75) over title+body."""
    docs_tokens = {d[0]: tokenize(d[3] + " " + d[4]) for d in DOCS}
    n = len(docs_tokens)
    avgdl = sum(len(t) for t in docs_tokens.values()) / n
    df: dict[str, int] = {}
    for toks in docs_tokens.values():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    scores: dict[str, float] = {}
    for doc_id, toks in docs_tokens.items():
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for term in set(tokenize(query)):
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            f = tf[term]
            s += idf * (f * 2.5) / (f + 1.5 * (1 - 0.75 + 0.75 * len(toks) / avgdl))
        scores[doc_id] = s
    return scores


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    def dcg(ids: list[str]) -> float:
        return sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 2)
                   for i, d in enumerate(ids[:k]))
    ideal = sorted(grades, key=lambda d: (-grades[d], d))
    denom = dcg(ideal)
    return dcg(ranked_ids) / denom if denom else 1.0
