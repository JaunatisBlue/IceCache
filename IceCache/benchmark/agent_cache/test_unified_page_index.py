import numpy as np
from icecache.retrieval import (
    CandidatePlan,
    CandidateQueryResult,
    DCIObserverGroup,
    query_candidate_provider,
    validate_leaf_ids,
)
from icecache.materialization import CandidatePlan as MaterializationCandidatePlan

from dci_profile_cost_calibration import (
    aligned_page_costs,
    parse_profile_line,
    usable_recall_profiles,
)
from recall_cost_trace_probe import summarize as summarize_recall_trace
from chance_constraint_calibration import (
    aggregate_decode_steps,
    calibrate,
    conformal_upper_residual,
)

from icecache.experimental import (
    ExactLeafCentroidProvider,
    LayerDCIBackend,
    ShadowJointSelector,
    RuntimeDescriptorTrace,
    PageCandidate,
    SemanticPage,
    SessionMaterialization,
    LayeredPartitionedIndex,
    UnifiedPageCatalog,
    select_pages,
    select_pages_exact,
)


def page(
    page_id, source, descriptor, costs, session_id=None, error=0.0, layer=0, head=0
):
    return SemanticPage(
        page_id=page_id,
        source=source,
        descriptor=np.asarray(descriptor),
        error_radius=error,
        load_cost_samples=np.asarray(costs),
        session_id=session_id,
        layer_idx=layer,
        head_idx=head,
    )


def test_search_mixes_sources_but_filters_other_sessions():
    catalog = UnifiedPageCatalog()
    catalog.extend(
        [
            page("rag", "rag", [1, 0], [1, 1, 1]),
            page("own", "prompt", [0.9, 0.1], [1, 1, 1], "a"),
            page("foreign", "prompt", [1, 0], [1, 1, 1], "b"),
        ]
    )
    found = catalog.search(np.array([1, 0]), session_id="a", candidate_limit=10)
    assert {item.page.page_id for item in found} == {"rag", "own"}
    assert catalog.remove_session("a") == 1
    assert catalog.source_counts() == {"prompt": 1, "rag": 1}
    assert {item.page.page_id for item in catalog.search(
        np.array([1, 0]), session_id="a", candidate_limit=10
    )} == {"rag"}


def test_catalog_metadata_is_immutable_and_filtered_ids_fail_closed():
    descriptor = np.array([1.0, 0.0])
    costs = np.array([1.0, 2.0])
    item = page("rag", "rag", descriptor, costs)
    catalog = UnifiedPageCatalog()
    catalog.add(item)
    descriptor[0] = 0.0
    costs[0] = 99.0
    assert item.descriptor.tolist() == [1.0, 0.0]
    assert item.load_cost_samples.tolist() == [1.0, 2.0]
    assert not item.descriptor.flags.writeable
    assert not item.load_cost_samples.flags.writeable
    try:
        item.descriptor.setflags(write=True)
    except ValueError:
        pass
    else:
        raise AssertionError("descriptor metadata could be made writable again")
    try:
        catalog.search(
            np.array([1, 0]),
            session_id="s",
            candidate_limit=1,
            page_ids=["rag", "stale"],
        )
    except KeyError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("unknown page filter was silently ignored")


def test_rag_materialization_cost_is_session_local():
    catalog = UnifiedPageCatalog()
    catalog.add(page("rag", "rag", [1, 0], [99, 99, 99]))
    costs = {
        "descriptor_only": np.array([8, 10, 12]),
        "cpu_kv": np.array([2, 3, 4]),
        "gpu_kv": np.array([0, 0, 0]),
    }
    session_a = SessionMaterialization(costs, session_id="a")
    session_b = SessionMaterialization(costs, session_id="b")
    session_a.set_state("rag", "gpu_kv")
    candidate_a = catalog.search(
        np.array([1, 0]), session_id="a", candidate_limit=1, cost_resolver=session_a.resolve
    )[0]
    candidate_b = catalog.search(
        np.array([1, 0]), session_id="b", candidate_limit=1, cost_resolver=session_b.resolve
    )[0]
    assert candidate_a.materialization_state == "gpu_kv"
    assert candidate_b.materialization_state == "descriptor_only"
    assert candidate_a.load_cost_samples.max() == 0
    assert candidate_b.load_cost_samples.min() == 8
    try:
        catalog.search(
            np.array([1, 0]),
            session_id="b",
            candidate_limit=1,
            cost_resolver=session_a.resolve,
        )
    except ValueError as error:
        assert "session 'a'" in str(error) and "'b'" in str(error)
    else:
        raise AssertionError("session A materialization resolver leaked into session B")


def test_layered_partitioned_index_never_mixes_layer_or_session_spaces():
    index = LayeredPartitionedIndex([0, 4])
    index.add(page("rag-l0", "rag", [1, 0], [1, 1], layer=0))
    index.add(page("rag-l4", "rag", [0, 1], [1, 1], layer=4))
    index.add(page("a-l0", "prompt", [0.9, 0.1], [1, 1], "a", layer=0))
    index.add(page("b-l0", "prompt", [1, 0], [1, 1], "b", layer=0))
    result = index.search_layers(
        {0: np.array([1, 0]), 4: np.array([0, 1])},
        session_id="a",
        candidate_limit=10,
    )
    assert {item.page.page_id for item in result[0][0]} == {"rag-l0", "a-l0"}
    assert {item.page.page_id for item in result[4][0]} == {"rag-l4"}
    assert all(
        item.page.layer_idx == layer and item.page.head_idx == head
        for layer, heads in result.items()
        for head, items in heads.items()
        for item in items
    )
    assert index.remove_session("a") == 1
    assert {item.page.page_id for item in index.search_head(
        0, 0, np.array([1, 0]), session_id="a", candidate_limit=10
    )} == {"rag-l0"}


def test_single_catalog_rejects_cross_layer_page():
    catalog = UnifiedPageCatalog(layer_idx=0)
    try:
        catalog.add(page("wrong-layer", "rag", [1, 0], [1, 1], layer=4))
    except ValueError as error:
        assert "layer 4" in str(error)
    else:
        raise AssertionError("cross-layer page was accepted")


def test_single_catalog_rejects_cross_head_page():
    catalog = UnifiedPageCatalog(layer_idx=0, head_idx=0)
    try:
        catalog.add(page("wrong-head", "rag", [1, 0], [1, 1], head=1))
    except ValueError as error:
        assert "head 1" in str(error)
    else:
        raise AssertionError("cross-head page was accepted")


def test_selector_respects_joint_tail_budget_and_prompt_floor():
    catalog = UnifiedPageCatalog()
    catalog.extend(
        [
            page("prompt", "prompt", [0.8, 0.2], [2, 2, 4, 4], "a"),
            page("rag-cheap", "rag", [1, 0], [1, 1, 1, 1]),
            page("rag-tail", "rag", [1, 0], [1, 1, 8, 8]),
        ]
    )
    candidates = catalog.search(np.array([1, 0]), session_id="a", candidate_limit=10)
    result = select_pages(
        candidates,
        max_pages=3,
        resource_budget=5,
        confidence=0.75,
        min_prompt_pages=1,
    )
    ids = {item.page.page_id for item in result.selected}
    assert ids == {"prompt", "rag-cheap"}
    assert result.cost_quantile <= 5
    assert result.source_counts == {"prompt": 1, "rag": 1}
    assert result.feasible


def test_error_certificate_changes_ranking():
    catalog = UnifiedPageCatalog()
    catalog.extend(
        [
            page("risky-error", "rag", [1, 0], [1, 1, 1], error=0.5),
            page("bounded-error", "rag", [0.9, 0.1], [1, 1, 1], error=0.0),
        ]
    )
    found = catalog.search(
        np.array([1, 0]), session_id="a", candidate_limit=2, error_weight=1.0
    )
    assert found[0].page.page_id == "bounded-error"


def test_grouped_queries_rank_by_best_q_head_for_one_kv_head():
    catalog = UnifiedPageCatalog()
    catalog.extend(
        [
            page("first-q", "rag", [1, 0], [1, 1]),
            page("second-q", "rag", [0, 1], [1, 1]),
            page("irrelevant", "rag", [-1, 0], [1, 1]),
        ]
    )
    found = catalog.search(
        np.array([[1, 0], [0, 1]]), session_id="a", candidate_limit=3
    )
    assert {item.page.page_id for item in found[:2]} == {"first-q", "second-q"}
    assert found[-1].page.page_id == "irrelevant"


def test_runtime_descriptor_trace_builds_leaf_centroids():
    class FakeDB:
        token2node = (
            np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int32),
            np.zeros((2, 4), dtype=np.int32),
        )
        num_leaves = np.array([2, 2], dtype=np.int32)

    class FakeState:
        n_kv_heads = 2
        head_dim = 2
        seq_len = 9
        dci_db = [FakeDB()]
        selected_page_idx = [None]

        def check_reuse(self, _layer):
            return 0

        def _DCI_first_call(self, *_args):
            return None

        def _DCI_add(self, *_args):
            return None

        def _DCI_query(self, _b, layer, _queries):
            self.selected_page_idx[layer] = np.array([[0, 1], [1, 0]])
            return None

    state = FakeState()
    state.dci_observer = None
    trace = RuntimeDescriptorTrace(state)
    trace.install()
    keys = np.array(
        [
            [[1, 0], [1, 0], [0, 2], [0, 2]],
            [[3, 0], [0, 1], [0, 1], [0, 1]],
        ],
        dtype=np.float32,
    )
    trace.on_dci_event("build", layer_idx=0, key_states=keys)
    state.selected_page_idx[0] = np.array([[0, 1], [1, 0]])
    trace.on_dci_event(
        "query",
        layer_idx=0,
        query_states=np.ones((4, 2)),
        selected_leaf_ids=state.selected_page_idx[0],
    )
    snapshot = trace.layer_snapshot(0)
    np.testing.assert_allclose(snapshot["descriptors"][0, 0], [1, 0])
    np.testing.assert_allclose(snapshot["descriptors"][0, 1], [0, 1])
    assert snapshot["member_counts"].tolist() == [[2, 2], [1, 3]]
    assert len(trace.queries[0]) == 1


def test_candidate_provider_leaf_ids_are_validated_before_native_bookkeeping():
    valid = validate_leaf_ids(
        np.array([[0, 2], [1, 3]]),
        n_kv_heads=2,
        num_neighbours=2,
        num_leaves=np.array([3, 4]),
    )
    assert valid.dtype == np.int32 and valid.flags.c_contiguous
    for invalid in (
        np.array([[0, 0], [1, 2]]),
        np.array([[0, 3], [1, 2]]),
        np.array([[0, -1], [1, 2]]),
    ):
        try:
            validate_leaf_ids(
                invalid,
                n_kv_heads=2,
                num_neighbours=2,
                num_leaves=np.array([3, 4]),
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid provider result was accepted: {invalid}")


def test_candidate_plan_contract_has_one_canonical_type():
    assert CandidatePlan is MaterializationCandidatePlan


def test_plan_aware_provider_dispatch_is_atomic_and_legacy_compatible():
    effective = np.array([[0, 1]], dtype=np.int32)

    class Planned:
        def query_candidate_result(self, **_kwargs):
            return CandidateQueryResult(effective_leaf_ids=effective, plan="plan")

        def query_leaf_ids(self, **_kwargs):
            raise AssertionError("legacy path should not run")

    class Legacy:
        def query_leaf_ids(self, **_kwargs):
            return effective

    planned = query_candidate_provider(Planned(), layer_idx=0)
    legacy = query_candidate_provider(Legacy(), layer_idx=0)
    assert planned.effective_leaf_ids is effective and planned.plan == "plan"
    assert legacy.effective_leaf_ids is effective and legacy.plan is None


def test_candidate_provider_rejects_leaf_ids_that_alias_during_int32_cast():
    for invalid in (
        np.array([[0, 2**32], [1, 2]], dtype=np.uint64),
        np.array([[0, -(2**32)], [1, 2]], dtype=np.int64),
    ):
        try:
            validate_leaf_ids(
                invalid,
                n_kv_heads=2,
                num_neighbours=2,
                num_leaves=np.array([3, 4]),
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"int32-aliasing leaf was accepted: {invalid}")


def test_exact_leaf_centroid_provider_observes_dci_topology_and_groups_q_heads():
    class State:
        n_kv_heads = 2
        n_qo_heads = 4
        head_dim = 2

    class DB:
        token2node = (
            np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32),
            np.zeros((2, 4), dtype=np.int32),
        )
        num_leaves = np.array([2, 2], dtype=np.int32)

    keys = np.array(
        [
            [[1, 0], [1, 0], [0, 1], [0, 1]],
            [[0, 1], [0, 1], [1, 0], [1, 0]],
        ],
        dtype=np.float32,
    )
    provider = ExactLeafCentroidProvider()
    provider.on_dci_event(
        "build", state=State(), layer_idx=3, key_states=keys, dci_db=DB()
    )
    queries = np.array([[1, 0], [0.8, 0.2], [0, 1], [0.2, 0.8]])
    result = provider.query_leaf_ids(
        layer_idx=3, query_states=queries, num_neighbours=2, state=State()
    )
    assert result.tolist() == [[0, 1], [0, 1]]


def test_dci_page_backend_validates_native_layout_and_counts():
    class FakeIndex:
        num_leaves = np.array([3, 3], dtype=np.int32)

        def __init__(self, raw, counts):
            self.raw = np.asarray(raw, dtype=np.int64)
            self.counts = np.asarray(counts, dtype=np.int32)

        def query(self, *_args, **_kwargs):
            return self.raw, self.counts

    backend = object.__new__(LayerDCIBackend)
    backend.heads = [0, 1]
    backend.dimension = 2
    backend.num_points = 4
    backend.min_leaves = 3
    backend.parallel_level = 0
    backend.page_ids = {
        0: ["h0-0", "h0-1", "h0-2", "h0-3"],
        1: ["h1-0", "h1-1", "h1-2", "h1-3"],
    }
    # Per query row: [two leaf IDs | two point IDs].  The second query has
    # count=1, so its final point slot must never be consumed.
    backend.index = FakeIndex(
        [0, 1, 2, 0, 1, 0, 3, 2],
        [2, 1],
    )
    found = backend.query_page_ids(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        num_neighbours=2,
        prop_to_retrieve=1.0,
    )
    assert found == {0: ["h0-2", "h0-0"], 1: ["h1-3"]}

    backend.index = FakeIndex(
        # Swapped halves: point 3 is not a valid leaf (<3), so fail closed.
        [3, 0, 0, 1, 1, 0, 3, 2],
        [2, 1],
    )
    try:
        backend.query_page_ids(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            num_neighbours=2,
            prop_to_retrieve=1.0,
        )
    except RuntimeError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("swapped native DCI result halves were accepted")


def test_shadow_joint_selector_mixes_rag_and_real_prompt_leaf_descriptors():
    class State:
        n_kv_heads = 2
        n_qo_heads = 4
        head_dim = 2
        seq_len = 33

    class DB:
        token2node = (
            np.array([[0, 0], [0, 0]], dtype=np.int32),
            np.zeros((2, 2), dtype=np.int32),
        )
        num_leaves = np.array([1, 1], dtype=np.int32)

    provider = ExactLeafCentroidProvider()
    shadow = ShadowJointSelector(
        provider,
        session_id="session",
        prompt_cost_samples=np.array([1, 1, 1, 1]),
        max_pages=2,
        resource_budget=2,
        confidence=0.75,
        min_prompt_pages=1,
        rag_bias=0.2,
    )
    observers = DCIObserverGroup(provider, shadow)
    observers.on_dci_event(
        "build",
        state=State(),
        layer_idx=0,
        key_states=np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]]),
        dci_db=DB(),
    )
    shadow.register_rag_page(
        page("rag-h0", "rag", [1, 0], [1, 1, 1, 1], layer=0, head=0)
    )
    observers.on_dci_event(
        "query",
        state=State(),
        layer_idx=0,
        query_states=np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),
        selected_leaf_ids=np.array([[0], [0]]),
        recalled_leaf_ids=None,
        evicted_page_ids=None,
    )
    event = shadow.events[-1]
    assert event.feasible
    assert event.prompt_pages == 1 and event.rag_pages == 1
    assert "rag-h0" in event.selected_page_ids


def test_selector_preserves_minimum_width_for_every_kv_head():
    candidates = []
    for head, scores in ((0, (1.0, 0.9)), (1, (0.2,))):
        catalog = UnifiedPageCatalog(layer_idx=0, head_idx=head)
        for idx, score in enumerate(scores):
            catalog.add(
                page(
                    f"h{head}-{idx}",
                    "rag",
                    [score, (1 - score**2) ** 0.5],
                    [1, 1, 1],
                    head=head,
                )
            )
        candidates.extend(
            catalog.search(np.array([1, 0]), session_id="s", candidate_limit=4)
        )
    selected = select_pages(
        candidates,
        max_pages=2,
        resource_budget=2,
        confidence=0.95,
        min_pages_per_head=1,
    )
    exact = select_pages_exact(
        candidates,
        max_pages=2,
        resource_budget=2,
        confidence=0.95,
        min_pages_per_head=1,
    )
    assert selected.feasible and selected.head_floor_satisfied
    assert {item.page.head_idx for item in selected.selected} == {0, 1}
    assert selected.total_utility == exact.total_utility


def test_selector_handles_same_page_id_on_multiple_kv_heads():
    candidates = []
    for head in (0, 1):
        catalog = UnifiedPageCatalog(layer_idx=0, head_idx=head)
        catalog.add(page("shared", "rag", [1, 0], [1, 1], head=head))
        candidates.extend(
            catalog.search(np.array([1, 0]), session_id="s", candidate_limit=1)
        )
    result = select_pages(
        candidates,
        max_pages=2,
        resource_budget=2,
        confidence=0.95,
        min_pages_per_head=1,
    )
    assert result.feasible
    assert len(result.selected) == 2
    assert {item.page.head_idx for item in result.selected} == {0, 1}


def test_selector_never_exceeds_max_pages_when_floors_compete():
    candidates = []
    for head in (0, 1):
        for idx in range(2):
            candidates.append(
                PageCandidate(
                    page(f"rag-{head}-{idx}", "rag", [1, 0], [0, 0], head=head),
                    relevance=1.0,
                    utility=1.0,
                    load_cost_samples=np.array([0, 0]),
                    materialization_state="default",
                )
            )
    for idx in range(2):
        candidates.append(
            PageCandidate(
                page(f"prompt-{idx}", "prompt", [1, 0], [0, 0], "s", head=0),
                relevance=1.0,
                utility=1.0,
                load_cost_samples=np.array([0, 0]),
                materialization_state="default",
            )
        )
    result = select_pages(
        candidates,
        max_pages=4,
        resource_budget=0,
        confidence=0.8,
        min_pages_per_head=2,
        min_prompt_pages=2,
    )
    assert len(result.selected) <= 4
    assert not result.feasible
    assert not result.prompt_floor_satisfied


def test_selector_priority_uses_requested_confidence():
    low_median_high_tail = PageCandidate(
        page("tail", "rag", [1, 0], [1, 1, 1, 100, 100]),
        relevance=1.0,
        utility=10.0,
        load_cost_samples=np.array([1, 1, 1, 100, 100]),
        materialization_state="default",
    )
    stable = PageCandidate(
        page("stable", "rag", [1, 0], [5, 5, 5, 5, 5]),
        relevance=1.0,
        utility=8.0,
        load_cost_samples=np.array([5, 5, 5, 5, 5]),
        materialization_state="default",
    )
    median = select_pages(
        [low_median_high_tail, stable],
        max_pages=1,
        resource_budget=100,
        confidence=0.5,
        local_search=False,
    )
    tail = select_pages(
        [low_median_high_tail, stable],
        max_pages=1,
        resource_budget=100,
        confidence=0.95,
        local_search=False,
    )
    assert median.selected[0].page.page_id == "tail"
    assert tail.selected[0].page.page_id == "stable"


def test_exact_oracle_finds_better_combination_than_greedy():
    catalog = UnifiedPageCatalog()
    catalog.extend(
        [
            page("large", "rag", [1, 0], [6, 6, 6, 6]),
            page("small-a", "rag", [0.8, 0.6], [5, 5, 5, 5]),
            page("small-b", "rag", [0.8, -0.6], [5, 5, 5, 5]),
        ]
    )
    candidates = catalog.search(np.array([1, 0]), session_id="a", candidate_limit=3)
    greedy = select_pages(
        candidates,
        max_pages=2,
        resource_budget=10,
        confidence=0.95,
        local_search=False,
    )
    refined = select_pages(candidates, max_pages=2, resource_budget=10, confidence=0.95)
    exact = select_pages_exact(candidates, max_pages=2, resource_budget=10, confidence=0.95)
    assert exact.total_utility > greedy.total_utility
    assert {item.page.page_id for item in exact.selected} == {"small-a", "small-b"}
    assert refined.total_utility == exact.total_utility


def test_infeasible_prompt_floor_is_explicit():
    catalog = UnifiedPageCatalog()
    catalog.add(page("prompt", "prompt", [1, 0], [10, 10, 10], "a"))
    candidates = catalog.search(np.array([1, 0]), session_id="a", candidate_limit=1)
    result = select_pages_exact(
        candidates,
        max_pages=1,
        resource_budget=2,
        confidence=0.95,
        min_prompt_pages=1,
    )
    assert not result.feasible
    assert not result.prompt_floor_satisfied


def test_profile_costs_are_aligned_and_scale_with_head_pages():
    row = parse_profile_line(
        'prefix DCI_PROFILE {"recall_pages_per_token": 100, '
        '"recall_wait_ms_per_token": 5, "decode_tpot_ms": 20}'
    )
    assert len(usable_recall_profiles([row])) == 1
    one = aligned_page_costs([row], 1)
    four = aligned_page_costs([row], 4)
    np.testing.assert_allclose(four, one * 4)


def test_recall_trace_summary_preserves_event_tail():
    events = [
        {
            "recalled_head_pages": 2,
            "query_ms": 1,
            "recall_submit_ms": 1,
            "blocking_total_ms": 4,
            "blocking_minus_query_submit_ms": 2,
        },
        {
            "recalled_head_pages": 8,
            "query_ms": 2,
            "recall_submit_ms": 2,
            "blocking_total_ms": 20,
            "blocking_minus_query_submit_ms": 16,
        },
    ]
    summary = summarize_recall_trace(events)
    assert summary["events"] == 2
    assert summary["blocking_total_ms"]["max"] == 20
    assert summary["pages_vs_blocking_correlation"] > 0.99


def test_conformal_calibration_aggregates_layers_and_holds_out_test():
    events = []
    for step in range(80):
        for layer in (2, 5):
            pages = 2 + (step % 4)
            events.append(
                {
                    "decode_step": step,
                    "layer": layer,
                    "recalled_head_pages": pages,
                    "blocking_minus_query_submit_ms": 0.5 * pages + (step % 5) * 0.01,
                }
            )
    observations = aggregate_decode_steps([{"events": events}], "blocking_minus_query_submit_ms")
    assert len(observations) == 80
    assert observations[0]["layers"] == 2
    result = calibrate(observations, alpha=0.1, seed=3)
    assert result["split"] == {"train": 40, "calibration": 20, "test": 20}
    assert not result["vacuous_bound"]
    assert result["test"]["coverage"] >= 0.8


def test_conformal_bound_is_vacuous_when_95pct_sample_is_too_small():
    assert np.isinf(conformal_upper_residual(np.arange(10.0), alpha=0.05))
