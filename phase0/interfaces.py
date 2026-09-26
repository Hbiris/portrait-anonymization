"""Phase 0 extension points; no service or network inference implementation."""
from typing import Protocol, Any, Dict

class FaceAnonymizer(Protocol):
    def anonymize(self, face_bgr: Any) -> Any:
        ...

class IdentityValidator(Protocol):
    def validate(self, original_face: Any, candidate_face: Any) -> Dict:
        ...

class UtilityValidator(Protocol):
    """Independent of the landmark/state validator; missing evidence is not a pass."""
    def validate(self, original: Any, candidate: Any, context: Dict) -> Dict:
        ...

class DownstreamConsistencyEvaluator(Protocol):
    """Future locally hosted care-model evaluation on paired inputs, never implied implemented."""
    def compare(self, original: Any, candidate: Any, task_spec: Dict) -> Dict:
        ...

def zero_face_decision(detector_results, policy='reject'):
    """No detector can prove absence. 'continue' is only an explicit business policy."""
    if policy not in ('reject', 'continue_if_all_clear'):
        raise ValueError('Unknown zero-face policy')
    if len(detector_results) < 2:
        return 'reject_insufficient_detectors'
    if any(d.get('status') != 'ok' or d.get('suspected_faces', 0) != 0 for d in detector_results):
        return 'reject_face_suspected_or_detector_error'
    return 'continue_no_face_policy' if policy == 'continue_if_all_clear' else 'reject_zero_face_policy'
