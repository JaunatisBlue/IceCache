"""Exact page-centroid candidate provider for controlled runtime ablations."""

from __future__ import annotations

import threading

import numpy as np


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


class ExactLeafCentroidProvider:
    """Replace native candidate search with exact cosine over DCI leaf centroids.

    The object must be installed as both ``dci_observer`` and
    ``dci_candidate_provider``. It ranks only leaves already materialized by
    IceCache; consequently, page diff, CPU recall, and GPU scatter retain their
    original semantics. This is an ablation baseline, not yet a RAG injector.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.keys: dict[int, list[np.ndarray]] = {}
        self.descriptors: dict[int, np.ndarray] = {}
        self.num_leaves: dict[int, np.ndarray] = {}

    def on_dci_event(self, event: str, **payload) -> None:
        if event not in ("build", "add"):
            return
        layer = int(payload["layer_idx"])
        state = payload["state"]
        keys = _numpy(payload["key_states"])
        db = payload["dci_db"]
        with self.lock:
            self.keys.setdefault(layer, []).append(keys)
            all_keys = np.concatenate(self.keys[layer], axis=1)
            node_index, _node_offset = db.token2node
            node_index = np.asarray(node_index, dtype=np.int32).reshape(
                state.n_kv_heads, -1
            )
            if node_index.shape[1] != all_keys.shape[1]:
                raise ValueError(
                    f"layer {layer}: DCI maps {node_index.shape[1]} tokens, "
                    f"but provider observed {all_keys.shape[1]}"
                )
            leaves = np.asarray(db.num_leaves, dtype=np.int32).copy()
            centroids = np.zeros(
                (state.n_kv_heads, int(leaves.max()), state.head_dim),
                dtype=np.float32,
            )
            for head in range(state.n_kv_heads):
                for leaf in range(int(leaves[head])):
                    members = all_keys[head, node_index[head] == leaf]
                    if not members.size:
                        continue
                    centroid = members.mean(axis=0)
                    centroids[head, leaf] = centroid / max(
                        np.linalg.norm(centroid), 1e-12
                    )
            self.descriptors[layer] = centroids
            self.num_leaves[layer] = leaves

    def query_leaf_ids(
        self,
        *,
        layer_idx: int,
        query_states: np.ndarray,
        num_neighbours: int,
        state,
    ) -> np.ndarray:
        with self.lock:
            if layer_idx not in self.descriptors:
                raise RuntimeError(f"layer {layer_idx} has no centroid snapshot")
            descriptors = self.descriptors[layer_idx]
            leaves = self.num_leaves[layer_idx]
            queries = _numpy(query_states)
            expected = (state.n_qo_heads, state.head_dim)
            if queries.shape != expected:
                raise ValueError(f"queries have shape {queries.shape}, expected {expected}")
            queries /= np.maximum(
                np.linalg.norm(queries, axis=1, keepdims=True), 1e-12
            )
            ratio = state.n_qo_heads // state.n_kv_heads
            result = np.empty((state.n_kv_heads, num_neighbours), dtype=np.int32)
            for head in range(state.n_kv_heads):
                if num_neighbours > leaves[head]:
                    raise ValueError(
                        f"layer {layer_idx}, head {head} has only {leaves[head]} leaves"
                    )
                group = queries[head * ratio : (head + 1) * ratio]
                scores = group @ descriptors[head, : leaves[head]].T
                relevance = scores.max(axis=0)
                # Stable ordering makes ties deterministic across replays.
                order = np.argsort(-relevance, kind="stable")[:num_neighbours]
                result[head] = order.astype(np.int32, copy=False)
            return result
