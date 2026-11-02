"""LinearRAG components adapted for RepoDoctor; GPL-3.0, see LICENSE.txt.

Source: https://github.com/DEEP-PolyU/LinearRAG
Pinned commit: bcc94e66c221f798801255efba09311d6fbcd8d6
Modified by RepoDoctor from 2026-09-22; release cleanup on 2026-11-02.

core.py derives from upstream src/LinearRAG.py; embedding_store.py and utils.py
derive from the corresponding upstream modules. Local changes inject Java
entity/statement extraction and a windowed encoder, use literal identifier
boundaries and qualified seed matching, and retain strongest-path activation.
Cache changes add read-only loading, float32 batch IO, atomic persistence,
incremental vector reuse, graph recovery, stable ordering and progress events.
Unused upstream QA, vectorized propagation and attribute heuristics are omitted.
The strongest-path propagation is a local adaptation of the upstream algorithm.
"""
