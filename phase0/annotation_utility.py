"""Evaluate independent, visibility-aware annotations tied to exact image hashes.

This does not infer annotations, diagnose care risk, or substitute for downstream
AI testing. Consistent annotations alone never approve clinical utility/release.
"""
FIELDS = {
    'eye_state': {'both_open', 'both_closed', 'asymmetric', 'uncertain'},
    'mouth_state': {'closed', 'open', 'uncertain'},
    'facial_expression': {'neutral', 'smile', 'frown', 'other', 'uncertain'},
    'head_pose': {'frontal', 'turned', 'up', 'down', 'tilted', 'uncertain'},
}
VISIBILITY = {'visible', 'occluded', 'uncertain'}


def evaluate_annotations(review, source_sha256, candidate_sha256):
    pending = {'status':'indeterminate','pass':None,'release_allowed':False,
               'downstream_consistency':'not_implemented'}
    if (review.get('source_sha256') != source_sha256 or
            review.get('candidate_sha256') != candidate_sha256):
        return dict(pending, reason='review_image_hash_mismatch')
    if not review.get('reviewer') or review.get('review_type') not in ('research_observer','care_expert'):
        return dict(pending, reason='missing_review_provenance')
    changed, unknown = [], []
    for field, states in FIELDS.items():
        pair = review.get('fields',{}).get(field,{})
        original, candidate = pair.get('original',{}), pair.get('candidate',{})
        if any(item.get('state') not in states or item.get('visibility') not in VISIBILITY
               for item in (original,candidate)):
            unknown.append(field);continue
        if original['visibility'] != 'visible' or original['state']=='uncertain':
            if candidate['visibility']=='visible' and candidate['state']!='uncertain':
                changed.append(field+':generated_state_without_source_evidence')
            else:
                unknown.append(field)
        elif candidate['visibility']!='visible' or candidate['state']=='uncertain':
            unknown.append(field)
        elif candidate['state']!=original['state']:
            changed.append(field+':state_changed')
    if changed:
        return dict(pending,status='fail',**{'pass':False},reason='independent_observation_mismatch',
                    changed=changed,unknown=unknown)
    return dict(pending,reason='unobservable_or_missing_annotations' if unknown else 'observations_consistent_downstream_not_evaluated',
                changed=[],unknown=unknown)


def review_template(sample_id, arm, face_index, source_sha256, candidate_sha256):
    return {'sample_id':sample_id,'arm':arm,'face_index':face_index,
            'source_sha256':source_sha256,'candidate_sha256':candidate_sha256,
            'reviewer':None,'review_type':None,
            'fields':{name:{side:{'state':None,'visibility':None} for side in ('original','candidate')}
                      for name in FIELDS},'notes':None}
