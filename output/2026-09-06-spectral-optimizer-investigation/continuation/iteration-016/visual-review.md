# I16 inline visual follow-up review

Status: PASS

## Evidence and plotted data

- Accepted summary SHA-256:
  `393a2744df6fa97c9fdd4b684ba9a0e82a5fec8247253029ffe23e445cc30727`.
- Independent report-audit SHA-256:
  `52bada70c9e071fca20aa914d1d019f95b76bc55132256e0fc0ceb5dda3c8c18`.
- Plot source SHA-256:
  `16f837900428baccf1fee0700049da015b445622ad847aa3e495f0236575a17b`.
- Manifest SHA-256:
  `863f7a8ae8fb58826080e18161256db07223f97f793ffb1b12ecb77bfebdd7d5`.
- Learning-curves PNG SHA-256:
  `69a0c65ccbf89c510a53100598820f4f16eea1d0aed281baca0db4b986405d11`.
- Selection-effects PNG SHA-256:
  `2297f75b5f37202d76fd31a018803e2e3df7ea091f802682cd358b23ce1b274c`.

Main read the plotting source and inspected both final PNGs. The separate
`make_visual_summary.py --verify-only` invocation passed (terminal f5b66e).
Manifest figure membership was explicitly checked to be exactly these two PNGs.
The script verifies the pinned sources and passing exact report audit before
reading all 30 complete trajectories and all eight primary comparisons.

Learning curves retain all five policies, both training targets, all three
seeds and all six horizons. Each point is an equal-seed mean on the disjoint
clean auxiliary set. Accuracy is percent; CE is raw cross-entropy, lower better.
The x-axis uses actual numeric cumulative updates. Main caught a draft that
incorrectly spaced the unequal horizons equally; this was corrected before
delivery. The 250-update point remains at x=250, without a crowded major tick.

Selection effects retain every seed and mean, including the +1.653 pp
spectral-favorable mean and the mixed-seed near-zero CE comparison. Negative
effects favor scalar; CE utility is minus CE. Dots are individual seeds, not
confidence intervals. The HTML preserves unequal selection opportunities,
adaptive-panel limitations, provisional mechanism attribution and the fact
that the proposed normalized-gain experiment is not yet running. No universal
spectral-failure, equivalence or uniquely identified mechanism claim is made.