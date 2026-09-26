# Care Image Privacy Phase 0

Local GANonymization feasibility research for care images. This repository does
not implement a production service, FastAPI, OCR, or full-image face composition.
All generated candidates are unapproved research artifacts. No release is enabled.

## Current evidence

- [Implementation plan](Care_Image_Privacy_Implementation_Brief_EN.docx)
- [Initial Phase 0 report](phase0/REPORT.md)
- [Public-image attribution](phase0/ATTRIBUTION.md)
- [Python 3.8 inference dependencies](phase0/requirements.lock.txt)

The initial run used official GANonymization weights, YuNet cropping, SFace identity
measurements, and MediaPipe FaceMesh state measurements. Preprocessing differs from
the official RetinaFace pipeline. The next experiment controls that difference.

## Local artifacts

Models, vendor code, virtual environments, downloaded images, generated galleries,
and raw experimental outputs are excluded from Git. The current local environment
is `phase0/.venv`; `phase0/data/samples.json` identifies the 16 smoke-test inputs.
These are public photographs and explicitly labeled synthetic stress variants,
not a clinical validation dataset. Model and training-data commercial permissions
remain unresolved.

From a configured local checkout:

```sh
phase0/.venv/bin/python -m unittest discover -s phase0 -p 'test_spike.py' -v
```

Some initial evidence tests require the local downloaded data and prior results.
Inference on macOS also requires MediaPipe to create a local graphics context.

## Commit policy

Keep baseline evidence, experiment code, validation fixes, and conclusions in
separate commits. Do not commit weights, input/output images, environments or caches.
Record reproducible configurations and compact numeric evidence without claiming
that low identity similarity proves privacy or that landmarks prove care utility.
