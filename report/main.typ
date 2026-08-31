#import "arkheion-sans.typ": arkheion

#set page(
  background: align(center + horizon)[
    #rotate(45deg)[
      #text(size: 42pt, weight: "bold", fill: luma(235))[TikTok TechJam 2026]
    ]
  ],
)
#show: arkheion.with(
  title: "Kopibara RLab Agent: An autonomous ML researcher that searches over executable hypotheses, one measured code change at a time.",
  authors: (
    (name: "Phanuphat Srisukhawasu", email: "phanuphat.srisukhawasu@gmail.com", affiliation: "Kopibara Team"),
    (name: "Supachod Trakansirorut", email: "spchdt@gmail.com", affiliation: "Kopibara Team"),
  ),
  date: "September 1, 2026",
)

#set text(size: 10pt)
#show link: underline
#show math.equation: set text(font: "Fira Math")
#show raw: set text(size: 9pt)
#show raw.where(block: true): block.with(
  fill: luma(245), inset: 8pt, radius: 3pt, width: 100%,
)

= Objective and evaluation

The task is to automate the engineering loop for a recommendation ranker: inspect the
benchmark contract, construct features, train and tune a model, evaluate on public
validation data, and revise the code. KuaiRand ranks logged impressions within each
user. The relevance label is `long_view`, and the primary score is the mean of GAUC and
nDCG#text("@")5. KuaiRand-Pure is the required benchmark; KuaiRand-1K is the larger bonus
variant.

The controller develops against train and validation splits only. Test rows are used
only to produce the final score file, after their labels and feedback have been masked.
The hidden split is never read during search. The organizer's convergence rule is an
improvement of more than 0.002 over three consecutive iterations, with 50 iterations
and six hours as backstops.

The official Pure Factorization Machine reference is 0.6016 on validation primary.
For 1K, the starter kit does not publish an official reference, so the run measures
the supplied FM pipeline at 0.6079 and labels it as a runtime reference. This distinction
matters: the Pure comparison is against the organizer's published number, while the 1K
comparison is only an internal benchmark.

= Agent design

The controller uses `gpt-5.6-luna` with low reasoning effort as its planner. Its design
follows the executable code-search framing of #link("https://arxiv.org/abs/2502.13138")[AIDE]
and the reproducible benchmark discipline represented by #link("https://arxiv.org/abs/2410.07095")[MLE-bench].
At each turn, the planner receives the benchmark contract, the current solution tree,
the best measured node, and the source of the node selected as parent. It returns one
testable hypothesis and a small JSON list of exact-match replacements.

The runner applies a replacement only when its anchor occurs exactly once. The planner
may edit `history_lgbm.py` only, with at most four replacements, and the candidate must
retain the existing command-line interface and validation output. A denylist rejects
new network, shell, dynamic-import, or code-evaluation paths. Candidate processes run
with a copied environment that removes `OPENAI_API_KEY`, so the language-model
credential cannot reach generated code.

Each candidate is compiled and run in a bounded subprocess. The runner parses the
candidate's validation metrics, falls back to its `metrics.json` when stdout is
incomplete, and stores the hypothesis, exact diff, metrics, token counts, command,
and recovery events. The controller keeps the highest-scoring measured node as the
next parent. A failed candidate gets one repair request; a second failure is recorded
as `recovered_failure` and cannot become a parent.

= Seed pipeline

The seed is a grouped LightGBM ranker. It combines categorical context fields for the
user, video, author, tab, duration bucket, hour, random-exposure flag, date, 15-minute
time bucket, weekday, and released extra fields with continuous duration and time
features. The ranking groups are users, matching the evaluation unit rather than
treating the dataset as an ungrouped binary-classification problem.

The main inductive bias is a chronological history feature builder. For each user,
video, author, and user-video pair, it records prior interaction count, feedback rates,
log-transformed cumulative feedback, the last-seen timestamp transform, and the log
time since the previous observation. All released feedback signals are available to
these histories as auxiliary signals, while `long_view` remains the only scored label.
The state is updated only after an observed train or validation row, so a row cannot
use its own outcome or a later outcome. Test rows receive features but do not update
the state.

= Search results

The selected validation results are summarized in @results. The reported 1K primary
improves by 0.1419 over its measured FM reference; it is not an official hidden-test
delta. The Pure primary improves by 0.0283 over the organizer's official validation
reference.

#figure(
  table(
    columns: (1.15fr, 1.35fr, 0.95fr, 0.95fr, 0.95fr, 0.8fr),
    stroke: 0.5pt + luma(205),
    inset: (x: 8pt, y: 4pt),
    align: (left, right, right, right, right, right),
    table.header[*Benchmark*][*Reference*][*GAUC*][*nDCG#text("@")5*][*Primary*][*Delta*],
    [KuaiRand-Pure], [0.6016 official], [0.7059], [0.5538], [*0.6299*], [*+0.0283*],
    [KuaiRand-1K], [0.6079 measured], [0.7107], [0.7888], [*0.7498*], [*+0.1419*],
  ),
  caption: [Best validation results. The 1K reference is measured locally, not organizer-published.]
) <results>

#v(1em)

The Pure seed was already the best measured node at 0.6299. The search then checked
three children: a planned XE-NDCG change that was not effective because the fixed
Pure command still passed `--objective lambdarank`, truncation 10, and disabled query
normalization. Their primary scores were 0.6299, 0.6264, and 0.6268 respectively.
The no-op branch is retained in the run log as an execution mismatch rather than being
presented as evidence for or against XE-NDCG.

The 1K run found a much larger improvement. The seed scored 0.6038, below the measured
FM reference. Enabling grouped `rank_xendcg` raised primary to 0.6920, with nDCG#text("@")5
increasing to 0.6971. Increasing `num_leaves` from 31 to 63 then raised GAUC to
0.7034 and nDCG#text("@")5 to 0.7588, for primary 0.7311. A latest-feedback feature extension,
smaller leaves, path smoothing, and truncation levels 10 and 3 were all tested from
that parent and rejected.

The next accepted change added `lambda_l2=2.0`, moving primary to 0.7487. The planner
then tested stronger, weaker, and intermediate L2, more boosting rounds, a slower
learning rate, L1, feature subsampling, and categorical smoothing. Every one was lower
than the L2=2.0 parent or unchanged. Finally, `max_bin=511` improved primary to
0.7493. `max_bin=1023` fell to 0.7473, `max_bin=640` reached 0.7475, and the
intermediate `max_bin=767` became the best node at 0.7498.

#v(0.9em)
#figure(
  block(height: 215pt, clip: true)[
    #image("score-chart.jpg", width: 100%)
  ],
  caption: [1K validation trajectory. The lines show primary, GAUC, nDCG#text("@")5, and the measured reference.]
) <trajectory>

#v(1em)

The trajectory in @trajectory shows two different regimes. The objective change and
tree capacity produced the large gains; once the ranker reached 0.7311, the search
became local regularization and quantization tuning. This is the useful behavior of
the agent on this task: it moved from a model-family decision to narrow, measured
changes, kept only improvements, and preserved the best checkpoint when later trials
failed or regressed.

= Complete decision record

The full 1K sequence is below. The primary column is sufficient to explain parent
selection because the controller compares the mean score; the per-metric values remain
in each JSON iteration log. “Rejected” means the candidate ran but did not exceed the
current best. “Failed” means both the original candidate and its repair timed out.

#figure(
  text(size: 8.5pt)[
    #table(
      columns: (0.35fr, 2.85fr, 0.7fr, 0.8fr),
      stroke: 0.4pt + luma(215),
      inset: (x: 3pt, y: 1.5pt),
      align: (left, left, right, left),
      table.header[*Iter.*][*Decision tested*][*Primary*][*Outcome*],
      [0], [Seed: context plus four lagged multi-feedback histories], [0.6038], [Kept],
      [1], [Use grouped `rank_xendcg` instead of LambdaRank], [0.6920], [Kept],
      [2], [Increase `num_leaves` from 31 to 63], [0.7311], [Kept],
      [3], [Add latest feedback at each history scope], [0.7192], [Rejected],
      [4], [Reduce `min_data_in_leaf` to 20], [0.7292], [Rejected],
      [5], [Add `path_smooth=20.0`], [0.7285], [Rejected],
      [6], [Increase ranking truncation to 10], [0.7311], [Rejected],
      [7], [Reduce ranking truncation to 3], [0.7311], [Rejected],
      [8], [Add `lambda_l2=2.0`], [0.7487], [Kept],
      [9], [Increase L2 to 4.0], [0.7465], [Rejected],
      [10], [Allow 600 boosting rounds], [0.7487], [Rejected],
      [11], [Reduce L2 to 1.0], [0.7427], [Rejected],
      [12], [Use learning rate 0.03 and 700 rounds], [0.7448], [Rejected],
      [13], [Use intermediate L2 value 2.5], [0.7466], [Rejected],
      [14], [Add mild L1 regularization], [0.7366], [Rejected],
      [15], [Use feature fraction 0.85], [0.7445], [Rejected],
      [16], [Smooth categorical splits with `cat_smooth=20.0`], [0.7487], [Rejected],
      [17], [Increase `max_bin` to 511], [0.7493], [Kept],
      [18], [Increase `max_bin` further to 1023], [0.7473], [Rejected],
      [19], [Use intermediate `max_bin=767`], [0.7498], [Kept],
      [20], [Tune near the optimum with `max_bin=640`], [0.7475], [Rejected],
      [21], [Reduce XE-NDCG sigmoid scale to 0.5], [--], [Failed: timeout; retained 019],
    )
  ],
  caption: [Complete 1K search sequence recorded by the autonomous controller.]
) <search-record>

#v(1em)

The Pure sequence is shorter and is shown separately in @pure-record. It reached the
organizer convergence rule after three children. The first child demonstrates why the
run logs matter: its planner hypothesis was sensible, but the actual fixed command
overrode the candidate's new default. The measured record makes that limitation
visible and keeps the final claim narrow.

#figure(
  table(
    columns: (0.45fr, 3.3fr, 0.8fr, 1fr),
    stroke: 0.5pt + luma(205),
    inset: (x: 6pt, y: 3pt),
    align: (left, left, right, left),
    table.header[*Iter.*][*Decision tested*][*Primary*][*Outcome*],
    [0], [Seed: grouped LightGBM with leakage-safe histories], [0.6299], [Kept],
    [1], [XE-NDCG support planned; command still used LambdaRank], [0.6299], [No-op; rejected],
    [2], [Increase ranking truncation to 10], [0.6264], [Rejected],
    [3], [Disable query-size lambda normalization], [0.6268], [Rejected],
  ),
  caption: [Pure search sequence. Scores are validation primary.]
) <pure-record>

= Reproducibility, safety, and accounting

The run artifacts preserve each hypothesis, exact code diff, validation metrics,
subprocess command, token count, and recovery event. The final row-aligned CSV passed
the starter-kit's submission checker: 170,588 rows for Pure and 4,132,081 rows for
1K. Both manifests record zero manual interventions and `hidden_test_access: false`.

#figure(
  table(
    columns: (1.25fr, 1.5fr, 0.65fr, 0.85fr, 0.85fr, 0.75fr, 0.9fr),
    stroke: 0.5pt + luma(205),
    inset: (x: 6pt, y: 3pt),
    align: (left, left, right, right, right, right, left),
    table.header[*Benchmark*][*Run ID*][*Iters.*][*Wall-clock*][*LLM tokens*][*Manual*][*Stop*],
    [KuaiRand-Pure], [20260831T111545Z], [3], [5m 5s], [12,901], [0], [Converged],
    [KuaiRand-1K], [20260831T124656Z], [21], [6h 7m], [115,212], [0], [Wall-clock cap],
  ),
  caption: [Resource and autonomy accounting from the final manifests.]
) <accounting>

#v(0.9em)

The 1K run also records one planner recovery after a response exceeded the four-edit
limit. At iteration 21, the agent proposed reducing the XE-NDCG sigmoid scale. The
candidate and its repair both timed out after 375.92 seconds, so the controller marked
the branch `recovered_failure`, retained node `019-child`, and did not replace the
submission checkpoint. This is a failure-handling event, not a hidden-test result.

The reported metrics are public-validation results. Hidden-test scores are unavailable
until organizer evaluation, so no hidden-test improvement is inferred. The Pure result
is comparable with the official reference; the 1K result is reported against the
measured runtime reference only. The source implementation, run logs, manifests,
models, scores, and submissions are included in the project archive.
