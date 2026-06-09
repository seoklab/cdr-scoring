from data_loading.structure_graph_dataset import (  # noqa: F401
    DEFAULT_CDR_RANGES,
    OnTheFlyStructureGraphDataset,
    StructureSample,
    attach_af3_ranking,
    collate_structure_graphs,
    discover_structure_samples,
    load_af3_ranking_scores,
    make_structure_graph_loader,
    make_structure_inference_loader,
    parse_cdr_ranges,
)

# Backward-compatible aliases for the first temporary inference version.
OnTheFlyStructureDataset = OnTheFlyStructureGraphDataset
discover_af3_samples = discover_structure_samples
load_ranking_scores = load_af3_ranking_scores
