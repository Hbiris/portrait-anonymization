"""Small, model-independent helpers for paired Phase 0 experiments."""
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def new_run_directory(root, name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be a single directory name')
    target = Path(root) / name
    target.mkdir(parents=False, exist_ok=False)
    return target


def bbox_iou(a, b):
    """Bounding boxes are xyxy in original full-image pixel coordinates."""
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def match_boxes(a, b, threshold=.4):
    """Conservative mutual-best spatial matching; ambiguous pairs stay unmatched.

    This is for research pairing, not an assertion that every face was detected.
    """
    if not a or not b:
        return []
    matrix = [[bbox_iou(x, y) for y in b] for x in a]
    matches = []
    for i, scores in enumerate(matrix):
        best = max(scores)
        if best < threshold or scores.count(best) != 1:
            continue
        j = scores.index(best)
        reverse = [row[j] for row in matrix]
        if reverse.count(max(reverse)) == 1 and reverse.index(max(reverse)) == i:
            matches.append({'a': i, 'b': j, 'iou': best})
    return matches


def state_measurement_status(result):
    if result.get('status') != 'measured_uncalibrated':
        return 'indeterminate'
    pose = result.get('head_pose_difference_degrees')
    deltas = result.get('absolute_delta', {})
    required = ('ear_33_side', 'ear_263_side', 'mouth_aperture_ratio')
    values = [pose] + [deltas.get(key) for key in required]
    return ('measured_uncalibrated' if all(type(v) in (int, float) and math.isfinite(v) for v in values)
            else 'indeterminate')
