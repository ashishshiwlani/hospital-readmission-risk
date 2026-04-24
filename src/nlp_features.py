"""
Clinical NLP feature extraction for hospital readmission prediction.

Extracts embeddings from discharge summaries using two strategies:
1. Bio_ClinicalBERT: Domain-adapted transformer for clinical text (768-dim embeddings)
2. TF-IDF fallback: Lightweight keyword-based features for fast inference

Bio_ClinicalBERT is more powerful but slower; TF-IDF is useful for quick prototyping
or when GPU is unavailable.
"""

from typing import List
import numpy as np
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.feature_extraction.text import TfidfVectorizer


# Curated clinical vocabulary for TF-IDF fallback
# These terms have been manually selected for high correlation with readmission
CLINICAL_KEYWORDS = [
    "readmit",
    "follow",
    "concern",
    "monitor",
    "instable",
    "unstable",
    "pneumonia",
    "infection",
    "fever",
    "hypoxia",
    "hypotension",
    "heart",
    "respiratory",
    "kidney",
    "renal",
    "creatinine",
    "diabetes",
    "controlled",
    "poorly",
    "compliance",
    "medication",
]


class ClinicalNoteEmbedder:
    """
    Embed clinical discharge notes using Bio_ClinicalBERT.

    Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) is a BERT model fine-tuned
    on biomedical and clinical text from MIMIC-III. It produces 768-dimensional
    embeddings capturing semantic meaning of clinical text.

    Attributes:
        model_name (str): HuggingFace model identifier
        device (str): 'cuda' or 'cpu' for inference
        tokenizer: HuggingFace tokenizer
        model: HuggingFace transformer model
        batch_size (int): Batch size for inference
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: str | None = None,
        batch_size: int = 16,
    ):
        """
        Initialize the clinical note embedder.

        Args:
            model_name: HuggingFace model identifier (default Bio_ClinicalBERT).
            device: 'cuda' or 'cpu'. If None, auto-detects CUDA availability.
            batch_size: Number of notes to process in parallel for efficiency.
        """
        if not HAS_TORCH:
            raise ImportError(
                "torch and transformers required for ClinicalNoteEmbedder. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load Bio_ClinicalBERT from HuggingFace and set to eval mode.

        Prints model configuration for debugging (parameter count, embedding size).
        Models are cached locally in ~/.cache/huggingface after first download.
        """
        print(f"Loading {self.model_name} on device={self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Disable dropout, batch norm updates

        # Print model statistics for transparency
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Model loaded: {n_params / 1e6:.1f}M parameters, "
            f"embedding_dim=768, device={self.device}"
        )

    def embed_notes(self, notes: List[str]) -> np.ndarray:
        """
        Embed a batch of clinical notes.

        Processes notes in batches for memory efficiency. Uses mean-pooling over
        tokens to produce a single 768-dim vector per note:
            embed = mean(last_hidden_state over token dimension)

        Why mean-pooling vs CLS token?
        - CLS token is position-specific and trained on next-sentence prediction
        - Mean-pooling of all tokens captures full context without position bias
        - Better empirical performance on clinical text embeddings

        Args:
            notes: List of discharge summary texts (strings).

        Returns:
            np.ndarray of shape (n_notes, 768) with one row per note,
            all elements are float32.
        """
        embeddings = []

        # Process in batches to manage GPU/CPU memory
        for batch_start in tqdm(
            range(0, len(notes), self.batch_size),
            desc="Embedding notes",
        ):
            batch_end = min(batch_start + self.batch_size, len(notes))
            batch_notes = notes[batch_start:batch_end]

            # Tokenize with padding and truncation
            # max_length=512 is BERT's standard context window
            encoded = self.tokenizer(
                batch_notes,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Forward pass: get last hidden state (batch_size, seq_len, 768)
            with torch.no_grad():
                output = self.model(**encoded)
                last_hidden = output.last_hidden_state  # shape: (BS, seq_len, 768)

            # Mean-pool over token dimension to get (batch_size, 768)
            # attention_mask ensures padding tokens don't contribute to mean
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (BS, seq_len, 1)
            masked_hidden = last_hidden * attention_mask  # (BS, seq_len, 768)
            sum_hidden = masked_hidden.sum(dim=1)  # (BS, 768)
            sum_mask = attention_mask.sum(dim=1)  # (BS, 1)
            batch_embeddings = sum_hidden / sum_mask  # (BS, 768)

            embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate all batches
        return np.vstack(embeddings).astype(np.float32)

    def embed_single(self, note: str) -> np.ndarray:
        """
        Embed a single discharge note (convenience wrapper).

        Args:
            note: A single discharge summary string.

        Returns:
            np.ndarray of shape (768,) with embedding for the note.
        """
        embeddings = self.embed_notes([note])
        return embeddings[0]  # Return first (only) row


def extract_keyword_features(notes: List[str]) -> np.ndarray:
    """
    Extract lightweight clinical keyword features from discharge notes.

    Uses TF-IDF on a curated vocabulary of clinical terms known to correlate
    with readmission risk. Useful as a fast fallback when BERT is unavailable.

    Why TF-IDF vs raw bag-of-words?
    - TF (term frequency): penalizes overly frequent words (e.g., "patient")
    - IDF (inverse doc frequency): boosts rare discriminative words
    - Result: highlights clinically informative terms

    Args:
        notes: List of discharge summary texts.

    Returns:
        np.ndarray of shape (n_notes, len(CLINICAL_KEYWORDS)) with TF-IDF
        weights, sparse matrix converted to dense.
    """
    # Create TF-IDF vectorizer restricted to our curated vocabulary
    vectorizer = TfidfVectorizer(
        vocabulary=CLINICAL_KEYWORDS,
        lowercase=True,
        stop_words="english",
    )

    # Fit and transform notes to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(notes)

    # Convert sparse matrix to dense array for sklearn compatibility
    return tfidf_matrix.toarray().astype(np.float32)


if __name__ == "__main__":
    # Quick test: generate synthetic notes and extract embeddings
    test_notes = [
        "Patient admitted with pneumonia and sepsis. Course complicated by acute kidney injury. "
        "Creatinine elevated. Follow-up with infectious disease and nephrology crucial. "
        "Monitor closely for deterioration. Unstable at discharge.",
        "Routine coronary angiography performed. No acute findings. Patient stable. "
        "Discharge medications reviewed. Follow-up in one week.",
    ]

    print("Testing TF-IDF keyword features:")
    tfidf_features = extract_keyword_features(test_notes)
    print(f"TF-IDF shape: {tfidf_features.shape}")
    print(f"Example TF-IDF vector (note 0): {tfidf_features[0, :5]}")

    if HAS_TORCH:
        print("\nTesting Bio_ClinicalBERT embeddings:")
        embedder = ClinicalNoteEmbedder(batch_size=2)
        bert_embeddings = embedder.embed_notes(test_notes)
        print(f"BERT embedding shape: {bert_embeddings.shape}")
        print(f"Example BERT vector (note 0): {bert_embeddings[0, :5]}")
    else:
        print("\ntorch/transformers not installed; skipping BERT test")
