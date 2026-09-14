# Model files

Model files are deliberately kept inside this package. Nothing is installed
into the conda environment and `torch.hub` is disabled by the compatibility
port.

Expected paths:

- `rigidmask-sf/weights.pth`
- `midas/midas_v21_384.pt`

Both files retain their upstream licenses and download terms.

Verified SHA-256 values:

- RigidMask: `46c22a3432b4e2b61c0f69a160639ceed1ac33a167ae9ce1687d167b81d0e780`
- MiDaS v2.1: `f6b980704cfd7259c7cc2b058c2f160159c55e84b3b6c08331b4156a84629f70`
