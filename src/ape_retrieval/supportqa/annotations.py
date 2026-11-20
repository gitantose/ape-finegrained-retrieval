"""Validate and classify support annotations."""

def derive_support(proposal, known_ids):
    """Validate candidate sets; derive classification, never independent fact counts."""
    answerability = proposal.get('document_answerability', 'unreviewed')
    mapping = proposal.get('mapping_status', 'unreviewed')
    if answerability not in ('answerable', 'insufficient_document', 'ambiguous', 'unreviewed'):
        raise ValueError('Unknown document answerability')
    if mapping not in ('supported', 'decomposition_error', 'ambiguous', 'unreviewed', 'not_applicable'):
        raise ValueError('Unknown support mapping status')
    sets = proposal.get('support_sets', [])
    if any(not s or len(s) != len(set(s)) or not set(s) <= known_ids for s in sets):
        raise ValueError('Invalid or duplicate support proposition ID')
    sets = sorted({tuple(sorted(s)) for s in sets}, key=lambda s: (len(s), s))
    if any(set(a) < set(b) for a in sets for b in sets):
        raise ValueError('A proposed support set contains a sufficient proper subset')
    if sets and (answerability != 'answerable' or mapping != 'supported'):
        raise ValueError('Supports contradict answerability/mapping status')
    if mapping == 'supported' and not sets:
        raise ValueError('Supported mapping requires a candidate support set')
    if mapping == 'decomposition_error' and answerability != 'answerable':
        raise ValueError('Decomposition error requires an answerable document')
    sizes = [len(s) for s in sets]
    if sizes:
        label = 'single_support' if min(sizes) == 1 else 'multi_support'
    elif answerability == 'insufficient_document':
        label = 'insufficient_document'
    elif mapping == 'decomposition_error':
        label = 'decomposition_error'
    elif answerability == 'unreviewed' or mapping == 'unreviewed':
        label = 'unreviewed'
    else:
        label = 'ambiguous'
    return {'document_answerability': answerability, 'mapping_status': mapping,
            'support_sets': [list(s) for s in sets], 'support_cardinalities': sizes,
            'minimum_support_cardinality': min(sizes) if sizes else None,
            'support_type': label,
            'classification_rule': 'smallest candidate proposition-ID set; unrelated quality issues do not change cardinality'}
