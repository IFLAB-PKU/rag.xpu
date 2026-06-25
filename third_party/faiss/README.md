# Vendored FAISS (minimal)

This directory vendors the minimum FAISS artifacts required by `app/server/rag_pipeline.hpp`:

- `include/faiss/IndexFlat.h`
- `include/faiss/IndexFlatCodes.h`
- `include/faiss/Index.h`
- `include/faiss/MetricType.h`
- `lib/android-aarch64/libfaiss.so`
- `LICENSE`

Notes:
- We currently use `faiss::IndexFlatIP` for RAG searching.
- Header set is intentionally minimized to reduce repository size.
- If FAISS API usage changes, re-evaluate required headers.
