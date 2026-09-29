# Care Image Privacy Phase 0

Local GANonymization feasibility research for care images. This repository does
not implement a production service, FastAPI, OCR, or full-image face composition.
All generated candidates are unapproved research artifacts. No release is enabled.

## Current evidence

- [Implementation plan](Care_Image_Privacy_Implementation_Brief_EN.docx)
- [Latest preprocessing comparison](phase0/PREPROCESSING_REPORT.md)
- [Versioned measurements](phase0/evidence/preprocessing-v1/measurements.json)
- [Initial Phase 0 report](phase0/REPORT.md)
- [Public-image attribution](phase0/ATTRIBUTION.md)
- [Python 3.8 inference dependencies](phase0/requirements.lock.txt)

The initial run used official GANonymization weights, YuNet cropping, SFace identity
measurements, and MediaPipe FaceMesh state measurements. The completed comparison also runs upstream RetinaFace crops with alignment off/on.
The three arms produced 13/8/7 face candidates from 16 smoke-test inputs; none
are approved for release. See the latest report for denominator and utility limits.

## Local artifacts

Models, vendor code, virtual environments, downloaded images, generated galleries,
and raw experimental outputs are excluded from Git. The current local environment
is `phase0/.venv`; `phase0/data/samples.json` identifies the 16 smoke-test inputs.
These are public photographs and explicitly labeled synthetic stress variants,
not a clinical validation dataset. Model and training-data commercial permissions
remain unresolved.

From a configured local checkout:

```sh
phase0/.venv/bin/python -m unittest discover -s phase0 -p 'test_*.py' -v
```

Some initial evidence tests require the local downloaded data and prior results.
Inference on macOS also requires MediaPipe to create a local graphics context.

## Commit policy

Keep baseline evidence, experiment code, validation fixes, and conclusions in
separate commits. Do not commit weights, input/output images, environments or caches.
Record reproducible configurations and compact numeric evidence without claiming
that low identity similarity proves privacy or that landmarks prove care utility.

## Re-run the comparison in the configured local environments

Python 3.8.20 is used in both environments; do not combine their dependency locks.
The TensorFlow environment uses `tensorflow-macos`, whose distribution name differs
from the `tensorflow` requirement declared by `retina-face==0.0.13`. Install that
wrapper with `--no-deps` after installing the pinned TensorFlow/Keras/OpenCV/gdown
runtime; this is a platform packaging workaround, not permission to omit runtime
dependencies. The two lockfiles record the actual environments, not portable
cross-platform installation guarantees.

Required local assets: upstream source at `phase0/vendor/GANonymization-main`,
GAN weights at `phase0/models/publication-download`, `yunet.onnx` and `sface.onnx`
in `phase0/models`, RetinaFace weights at
`phase0/models/retinaface/.deepface/weights/retinaface.h5`, and the 16-image manifest
at `phase0/data/samples.json`. Source attribution is in `phase0/ATTRIBUTION.md`;
model URLs and hashes are in the reports. These assets are not included in Git.
A fresh clone requires downloading and verifying them before inference.

Use new run names; existing experiment directories are intentionally rejected:

```sh
phase0/.retina-venv/bin/python phase0/retina_preprocess.py --name retina-v2
phase0/.venv/bin/python phase0/compare_preprocessing.py --prepared retina-v2 --name preprocessing-v2
```

The first command executes local RetinaFace preprocessing. The second uses the
same generator and validators on all three arms. It needs the macOS graphics
context required by MediaPipe. Both save raw records under ignored `artifacts/`.
No images are uploaded. Neither command trains a model or starts a service.

To regenerate the current v1 gallery, compact CSV, full numeric evidence and blank
review template from the existing v1 artifacts:

```sh
phase0/.venv/bin/python phase0/export_preprocessing.py
```

For a different run call `export_preprocessing.export('preprocessing-v2')` from
within `phase0`. The gallery filename is reused; the numeric evidence is stored
under the named run. Preserve completed annotations under a different filename;
the exporter overwrites the blank review template. Never edit raw measured values
or turn missing measurements into passes.
