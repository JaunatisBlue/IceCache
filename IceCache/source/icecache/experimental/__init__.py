"""Experimental IceCache mechanisms that are not yet part of the stable API."""

from .dci_page_backend import LayerDCIBackend
from .centroid_provider import ExactLeafCentroidProvider
from .runtime_descriptor_trace import RuntimeDescriptorTrace
from .shadow_joint_selector import ShadowJointSelector, ShadowSelectionEvent
from .unified_page_index import (
    LayeredPartitionedIndex,
    PageCandidate,
    SelectionResult,
    SemanticPage,
    SessionMaterialization,
    UnifiedPageCatalog,
    select_pages,
    select_pages_exact,
)

__all__ = [
    "LayerDCIBackend",
    "ExactLeafCentroidProvider",
    "LayeredPartitionedIndex",
    "PageCandidate",
    "RuntimeDescriptorTrace",
    "ShadowJointSelector",
    "ShadowSelectionEvent",
    "SelectionResult",
    "SemanticPage",
    "SessionMaterialization",
    "UnifiedPageCatalog",
    "select_pages",
    "select_pages_exact",
]
