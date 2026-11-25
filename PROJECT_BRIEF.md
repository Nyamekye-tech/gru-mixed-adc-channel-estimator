# Project brief

Rebuild the BSc thesis study of a two-layer GRU estimator for a time-varying
single-user mixed-ADC Massive MIMO uplink as a rigorous, reproducible study.
The thesis setup uses 32 base-station antennas (4 high-resolution and 28
one-bit), a 2.6 GHz carrier, 0.5 ms pilot spacing, and user speeds from 3 to
130 km/h.

The original notebook and thesis are retained as legacy references. New
scientific code will be developed in `src/chest/` only after review and
approval of the corresponding stage.

## Research and reproducibility requirements

- Do not tune or weaken protocols to make the GRU win; report results honestly.
- Compare against meaningful classical baselines, including a two-scalar
  estimator and a quantization-aware Bussgang-LMMSE estimator.
- Use fair temporal input windows, explicit experiment configurations, and
  multiple seeds with uncertainty estimates.
- Save raw results with the configuration and source revision; generate
  figures and tables only from saved results.
- Keep large datasets, checkpoints, and raw outputs out of Git.
- Verify literature and citation details before using them.

## Stages

0. Review v1 and establish the project scaffold.
1. Implement and test channel/quantizer models and classical baselines.
2. Implement fair neural-model comparisons and tune each model within a
   documented, bounded search. The verified Gao et al. (2019) SIP-DNN is a
   single-slot selective-input model using high-resolution observations; do
   not conflate it with the thesis's unverified signal-reconstruction cascade.
3. Run the pre-registered full experiments and ablations.
4. Generate publication figures/tables from saved results and draft the paper
   around the results actually obtained.
