"""Capture real IceCache K/Q tensors and semantic-leaf membership for replay.

This module does not alter retrieval decisions. It wraps the three native DCI
entry points, records the tensors already crossing the Python/native boundary,
and exports page-centroid descriptors for offline ANN experiments. A centroid
is an experimental proxy; IceCache itself indexes tokens and forms leaves from
its projected ordering, so it must not be described as the native DCI score.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np


def _numpy(value, dtype=np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


class RuntimeDescriptorTrace:
    def __init__(self, state, *, layers: set[int] | None = None, max_queries: int = 0):
        self.state = state
        self.layers = layers
        self.max_queries = max_queries
        self.keys: dict[int, list[np.ndarray]] = {}
        self.queries: dict[int, list[dict]] = {}
        self.lock = threading.Lock()

    def _enabled(self, layer: int) -> bool:
        return self.layers is None or layer in self.layers

    def _db(self, layer: int):
        db = self.state.dci_db[layer]
        if db is None:
            db = self.state.dci_db[self.state.check_reuse(layer)]
        return db

    def install(self) -> None:
        if self.state.dci_observer not in (None, self):
            raise ValueError("InferState already has a DCI observer")
        self.state.dci_observer = self

    def on_dci_event(self, event: str, **payload) -> None:
        layer = int(payload["layer_idx"])
        if not self._enabled(layer):
            return
        if event in ("build", "add"):
            keys = _numpy(payload["key_states"])
            with self.lock:
                self.keys.setdefault(layer, []).append(keys)
            return
        if event != "query":
            return
        with self.lock:
            rows = self.queries.setdefault(layer, [])
            if self.max_queries > 0 and len(rows) >= self.max_queries:
                return
            selected = payload["selected_leaf_ids"]
            rows.append(
                {
                    "seq_len": int(self.state.seq_len),
                    "query": _numpy(payload["query_states"]),
                    "selected_leaf_ids": np.asarray(
                        selected, dtype=np.int32
                    ).copy(),
                }
            )

    def layer_snapshot(self, layer: int) -> dict[str, np.ndarray]:
        chunks = self.keys.get(layer, [])
        if not chunks:
            raise ValueError(f"layer {layer} has no captured keys")
        keys = np.concatenate(chunks, axis=1)
        node_index, _node_offset = self._db(layer).token2node
        node_index = np.asarray(node_index, dtype=np.int32).reshape(
            self.state.n_kv_heads, -1
        )
        if node_index.shape[1] != keys.shape[1]:
            raise ValueError(
                f"layer {layer}: captured {keys.shape[1]} tokens but DCI maps "
                f"{node_index.shape[1]} tokens"
            )
        num_leaves = np.asarray(self._db(layer).num_leaves, dtype=np.int32)
        max_leaves = int(num_leaves.max())
        descriptors = np.zeros(
            (self.state.n_kv_heads, max_leaves, self.state.head_dim), np.float32
        )
        counts = np.zeros((self.state.n_kv_heads, max_leaves), np.int32)
        for head in range(self.state.n_kv_heads):
            for leaf in range(int(num_leaves[head])):
                members = keys[head, node_index[head] == leaf]
                if members.size:
                    centroid = members.astype(np.float32).mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    descriptors[head, leaf] = centroid / max(norm, 1e-12)
                    counts[head, leaf] = len(members)
        return {
            "descriptors": descriptors,
            "member_counts": counts,
            "num_leaves": num_leaves,
            "node_index": node_index,
        }

    def export(self, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": 1,
            "descriptor": "normalized mean of post-RoPE keys in each DCI leaf",
            "warning": "proxy descriptor for offline evaluation; not native DCI scoring",
            "layers": {},
        }
        for layer in sorted(self.keys):
            snapshot = self.layer_snapshot(layer)
            query_rows = self.queries.get(layer, [])
            path = output_dir / f"layer-{layer}.npz"
            np.savez_compressed(
                path,
                **snapshot,
                queries=np.asarray([row["query"] for row in query_rows], dtype=np.float32),
                selected_leaf_ids=np.asarray(
                    [row["selected_leaf_ids"] for row in query_rows], dtype=np.int32
                ),
                seq_lens=np.asarray([row["seq_len"] for row in query_rows], dtype=np.int32),
            )
            manifest["layers"][str(layer)] = {
                "file": path.name,
                "tokens": int(snapshot["node_index"].shape[1]),
                "leaves_per_head": snapshot["num_leaves"].tolist(),
                "queries": len(query_rows),
            }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
