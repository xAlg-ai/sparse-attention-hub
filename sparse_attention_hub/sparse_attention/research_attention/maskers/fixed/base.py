"""Base fixed pattern masker implementations."""

from ..base import ResearchMasker


class FixedMasker(ResearchMasker):
    """Abstract base class for fixed pattern maskers."""

    pass


class TopKMasker(FixedMasker):
    """Abstract base class for top-K maskers."""

    pass


class TopPMasker(FixedMasker):
    """Abstract base class for top-P maskers."""

    pass 