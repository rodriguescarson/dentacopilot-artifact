# Public dental dataset loaders

Each loader downloads, caches, and parses one publicly available dental
dataset. We intentionally do *not* commit any image or chart data; loaders
write to `../cache/<dataset>/`, which is `.gitignore`-d.

| Dataset | Modality | Size | Access | Loader | Use in this paper |
|---|---|---|---|---|---|
| **DENTEX** (MICCAI 2023 grand-challenge) | panoramic X-ray | ~2 GB | research, registration required | `dentex.py` | imaging context for `M3` chart-grounding ablation |
| **DenPAR** (Sci Data 2025) | periapical X-ray | ~600 MB | open | `denpar.py` | local lesion features per tooth |
| **MMDental** (Sci Data 2025) | 3D CBCT + records | ~5 GB | CC-BY | `mmdental.py` | rare imaging-with-chart pairs |

## Why loaders are stubs (for now)

Two reasons we ship parsing scaffolding without auto-downloading:

1. **Licensing.** DENTEX and MMDental gate access behind a registration form
   or DUA. We will not bypass this; the loader prints the registration URL,
   asks the user to download manually, and verifies the SHA-256 of the
   resulting archive before caching.
2. **Disk + bandwidth.** A research workstation should not silently pull
   ~7 GB on `pip install`. The user runs `python loaders/<dataset>.py
   --download` explicitly when ready.

## Loader contract

Every loader exposes:

```python
def fetch(force: bool = False) -> pathlib.Path:
    """Download (interactive) and verify the dataset; return cache root."""

def load_split(split: str = "train") -> Iterable[Sample]:
    """Yield (image_path, annotations) pairs."""

def to_chart_context(sample: Sample, *, on_tooth: int | None = None) -> str:
    """Render a sample as a short, prompt-ready string describing the
    radiographic findings (used to supplement chart prefixes for M3)."""
```

The third method is what makes the loader useful for *this* paper: we use
imaging data not as a primary target but as additional chart context — a
prose-summary of "panoramic shows deep caries on tooth 19" that the LLM can
use to refine its next-procedure recommendation.
