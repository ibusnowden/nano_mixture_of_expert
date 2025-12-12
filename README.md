What you should look at right after merging this

exec/drop_rate_mean should be ~0 in your baseline sweeps.

exec/keep_rate_k1_mean / exec/keep_rate_k0_mean should be ~1 for K=2.

If k1 is lower, top-2 is silently degenerating.

exec/mass_min_min should move away from ~0 over time (otherwise “second expert never learns”).

If router entropy stays ~max but top1_margin never increases and mass stays flat, you’re stable but under-specializing.