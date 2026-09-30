"""Vendor-by-protocol grouping of MRI sessions (used for filtering and metric breakdowns)."""


def _unwrap_scalar(val, default=None):
    """DataLoader collation can wrap a subject scalar in a 1-element list/tuple; unwrap it."""
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) else default
    return val


def normalize_manufacturer(raw):
    """Map a raw DICOM Manufacturer string to a short tag: Siemens / Philips / GE / Other."""
    s = str(_unwrap_scalar(raw, '')).strip().lower()
    if not s or s in ('nan', 'none', 'unknown'):
        return 'Unknown'
    if 'siemens' in s:
        return 'Siemens'
    if 'philips' in s:
        return 'Philips'
    if s.startswith('ge') or 'general electric' in s:
        return 'GE'
    return 'Other'


def get_seq_group(subject):
    """Coarse T1w sequence group: MPRAGE / GRE_TFE (spoiled GRE) / SE / OTHER_*."""
    seq_type = subject.get('T1w_seq_type', 'GRE') if hasattr(subject, 'get') else 'GRE'
    seq_type = str(_unwrap_scalar(seq_type, 'GRE')).upper().strip()
    if 'MPRAGE' in seq_type:
        return 'MPRAGE'
    if 'GRE' in seq_type or 'TFE' in seq_type:
        return 'GRE_TFE'
    if 'SE' in seq_type:
        return 'SE'
    return f'OTHER_{seq_type}'


def manufacturer_seq_group(subject):
    """Return (manufacturer, seq_group, "<manufacturer>_<seq_group>") for a subject."""
    mfr = normalize_manufacturer(subject.get('manufacturer', 'Unknown')) if hasattr(subject, 'get') else 'Unknown'
    group = get_seq_group(subject)
    return mfr, group, f"{mfr}_{group}"
