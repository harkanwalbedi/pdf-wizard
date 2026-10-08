"""Source normalization: data files, documents, reference forms, and notes -> S0..Sn."""

from .common import LoadedSources, SourceInput
from .loader import load_sources

__all__ = ["LoadedSources", "SourceInput", "load_sources"]
