# V8 citation/source audit — 28 October 2026

The earlier reference existence/status audit remains in REFERENCE_AUDIT.md (26 October). This revision additionally checked the newly cited model references and the exact source paths supporting representation claims; it is not a new exhaustive semantic verification of every cited paper.

- GraphSAGE: Hamilton, Ying and Leskovec, Inductive Representation Learning on Large Graphs, NIPS 2017. Official metadata/paper: https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html . Metadata uses Zhitao Ying; the paper uses Rex Ying, retained in the bibliography.
- LightGBM: Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NIPS 2017. Title, eight authors and venue verified: https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html . No proceedings DOI found on the primary page.
- RelBench: actual vendor HEAD is 91c1f6420a03feb61c279aaeb194d7268435f278, with no modified tracked source. modeling/graph.py constructs FK-specific relation names and masks the requested feature columns. Executed results use its GraphSAGE and loader implementation in a reduced configuration, not leaderboard reproduction.
- RT: fetched def35f92bf890741c4be41c4c315886365ee5ffb. rustler/src/common.rs:22–27 defines edges with node index, table-name index, table type and timestamp; rustler/src/pre.rs:780–786 constructs forward edges without originating FK-column identity. Nodes retain other metadata. This supports the conditional structural collision risk only, not end-to-end role loss. Source: https://github.com/stanford-star/relational-transformer/blob/def35f92bf890741c4be41c4c315886365ee5ffb/rustler/src/common.rs#L22-L27 .
- Pinned task manifest bytes match SHA256 4c8be3afbaaa4bde9a98084cb1c11e390dda1c7e3219891cf2dc73381d3c424d. Materialization and data hashes are in results/v8/preparation/hashes.json.

Effect sizes were checked independently from raw predictions. The real-data interval includes zero and cannot substantiate a production-invariance claim. Final citation cross-references and PDF checks are under qa/v8/.
