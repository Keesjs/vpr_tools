# VPR Tools

Visual Place Recognition utility functions.

## Installation

```bash
pip install -r requirements.txt
from utils import cosine_similarity, find_top_k
cosine_similarity(a, b): Compute similarity between two vectors

batch_cosine_similarity(X): Compute NxN similarity matrix

find_top_k(query, database, k): Find top-k similar vectors
MIT
