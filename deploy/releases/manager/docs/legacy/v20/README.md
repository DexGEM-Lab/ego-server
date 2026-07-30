# Legacy V20 Documents

These files are historical V20 planning, runbook, and audit artifacts. They are preserved for reference and migration context only.

Current V21 execution should start from:

- `docs/parallel_annotation_draft.md`
- `docs/v21_english_orchestration.md`
- `docs/pipeline_v21_design.md`
- `docs/v21_run_contract.md`
- `docs/v21_component_extraction.md`

For V21 raw-video-to-segmentation, the active chain is:

```text
raw/source frame manifests
-> agent object plan
-> agent-selected OWLv2 detector keyframes
-> OWLv2 keyframe bbox proposals
-> approved OWLv2 bbox prompts
-> SAM2 proper full-video propagation
-> segmentation_sam2_proper contamination review
-> accepted masks for visible geometry
```
