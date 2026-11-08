# COCO val2017 local manifest

- Source: COCO official download server
- Images directory: `data/coco/val2017`
- Image count: 5,000 JPEG files
- Total image bytes: 814,705,164
- Annotation file: `data/coco/annotations/instances_val2017.json`
- Annotation bytes: 19,987,840
- Annotation SHA-256: `E8C7F7908F1D7278341FAE127D0DA654F102F11BD7B21D8AEEFA635B8C810B6F`
- Downloaded and verified: 2026-11-08 (Asia/Bangkok)

The original ZIP archives were removed after successful extraction. The benchmark reads the original JPEG files and creates corrupted images in memory.

Runtime verification: PyTorch 2.15.0 with CUDA 13.2 detects the NVIDIA GeForce RTX 4050 Laptop GPU (6 GB).

