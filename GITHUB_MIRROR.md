# GitHub mirror of `Jev-Style-Qwen3.5-2B-Decision-v2`

This repository mirrors the public Hugging Face model at:

<https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2>

Files smaller than 100 MiB are stored on the `main` branch. Larger model files are attached to this GitHub Release:

<https://github.com/lawrence3699/Jev-Style-Qwen3.5-2B-Decision-v2/releases/tag/huggingface-snapshot-2026-09-24>

See `RELEASE_ASSETS.tsv` for asset names, original paths, sizes, and SHA-256 checksums.

## Reassembling split files

Assets ending in `.part-aa`, `.part-ab`, and so on are consecutive pieces of one original file. Download every piece and concatenate them in lexical order. For example:

```bash
cat model.safetensors.part-* > model.safetensors
```

Verify the reconstructed file against the `original_sha256` value in `RELEASE_ASSETS.tsv`.
