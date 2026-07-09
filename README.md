# CSC401 – Assignment 2: Neural Machine Translation with Transformers

Individual assignment for University of Toronto's CSC401 (Natural Language Computing): implementing BLEU score evaluation and a from-scratch Transformer encoder-decoder for French→English translation on the Canadian Hansard corpus.

## Layout

- **`a2_bleu_score.py`** — n-gram precision, brevity penalty, and combined `BLEU_score` implementation.
- **`a2_dataloader.py`** — builds English/French vocabularies and a `HansardDataset` (Canadian parliamentary proceedings, paired `.e`/`.f` files) for training.
- **`a2_transformer_model.py`** — a Transformer implemented from scratch: `LayerNorm`, `MultiHeadAttention`, `FeedForwardLayer`, encoder/decoder layers and stacks, positional encoding, and the full `TransformerEncoderDecoder`.
- **`a2_transformer_runner.py`, `a2_main.py`** — training/evaluation entry points.
- **`a2_utils.py`** — shared helpers.
- **`test_a2_bleu_score.py`** — course-provided tests for the BLEU implementation.
- **`a2_sentences.txt`** — sample sentences used for translation comparison.
- **`a2_report/`** — the LaTeX report (`analysis.tex`/`.pdf`, `bonus.tex`, WandB training-curve figures).
- **`ID.txt`** — MarkUs submission identity declaration (not reproduced here).
- **`a2.pdf`** (in the parent folder) — the assignment handout.

## Report highlights (`a2_report/a2_report/analysis.tex`)

- Confirms BLEU-3 scores slightly higher than BLEU-4 on the same candidates, since longer n-grams are less likely to match the reference.
- Compares **pre-layer-norm** vs. **post-layer-norm** Transformer variants: pre-LN reaches a much higher BLEU (40.9 BLEU-3 / 48.0 BLEU-4) than post-LN (29.7 / 34.9) after the same training run.
- Compares translation quality across three systems on the same French sentences: the from-scratch Transformer (frequent `<unk>` tokens, weak on complex sentences), a T5 MT model (more fluent but occasionally loses nuance), and ChatGPT-4o (most fluent and contextually accurate).

## Author

Zixuan Zeng
