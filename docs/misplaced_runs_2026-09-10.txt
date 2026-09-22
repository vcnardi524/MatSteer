Runs that were written into the wrong property tree.

steered_test_alpha40.0_layer14 and steered_test_alpha80.0_layer14 are DENSITY steering
arms (linear, layer 14, sg family) that were generated with
--results-dir steering_results/energy_above_hull. They sat in the hull tree with no
relaxed CIFs, so hull predictions for them would have failed, and once predicted they
would have joined the hull family in steering_runs.csv -- the same cross-property
contamination that once paired density's nosg arms against the band-gap control.

They are NOT copies of the density-tree files of the same name: all 3,000 CIFs differ in
both, so these are separate generation runs. Moved rather than deleted for that reason.
Nothing scans this directory; PROPS names specific results dirs.

Moved 2026-09-10.
