# Inaccurate rule-based task rewards in IDEA AWM OPSD

Training `locaszzz/IDEA_sft` with OPSD on `locaszzz/IDEA_training_data` optimizes generated completion rules that sometimes award full reward to unsuccessful trajectories.

| | |
| --- | --- |
| **Author** | Jinyang Wu |
| **Domain** | reward |
| **Tags** | `idea` `verl` `awm` `opsd` `issue-only` |
| **Model** | locaszzz/IDEA_sft |
| **Dataset** | locaszzz/IDEA_training_data; locaszzz/AgentWorldModel-1K; 9,872 train / 128 validation tasks |
| **Framework** | IDEA / veRL / Agent World Model, `run_local.sh --train` |
| **Environment** | `idea-issue` · CPU-only diagnosis; no GPU or training required |
| **Evaluations** | issue |
| **Recorded runs** | 1 historical run, 0 corrected/control runs; 200 configured steps, W&B history covers 1-199 |

## Difficulty

The recorded `config/.hydra/config.yaml` selects `env.awm.reward_mode: code`.
In `idea_source/agent_system/environments/env_package/awm/bridge.py`, that mode
selects generated pure-code completion verifiers and maps their labels into rewards.
Some rules accept a different user's pre-existing dashboard, incidental identifier
matches, or seeded entities without checking the requested creation. Merely omitting
an LLM judge is not itself a defect; the incorrect acceptance rules are the cause.

The run reached tracked `critic/score/mean` values of 0.02455 at step 1 and
0.37402 at step 199. The frozen evidence contains 20 complete episodes, including
8 audited full-reward counterexamples across 7 tasks: 1 has archived database
evidence and 7 have recorded tool-history evidence plus separately labelled
seed-only verifier probes. This curated subset does not measure the population
error rate; all 8 cases are from the training split. The final configured
step 200 is absent from W&B history, and the last stdout record is truncated.
Neither rising reward nor that truncated record establishes semantic success or
a separate training failure.

## Reference solution

- **issue**: identify inaccurate rule-based completion rewards, name the configuration/code path, and connect a concrete verifier defect to a rewarded but unsuccessful trajectory. Do not require a patch, an LLM replacement, or a new training run. Detailed reviewer evidence is in [verifier resources](evaluations/issue/verifier/resources/README.md).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires verdict = 1.0, root cause > 0.8, and evidence > 0.8, matching the current benchmark judge.

Prepare downloads dataset inputs from `locaszzz/AgentWorldModel-1K` at commit
`6f8a98a043c631ba9c88dd1d1d1d1829a1f9dca5`, checking SHA-256 and byte sizes.
Frozen logs and trajectories ship with the task; diagnosis and verification are
offline after prepare. Offline checks replay the 8 cases, verify the 20-episode subset
and check answer isolation. Hosted issue-agent/judge attempts were performed on 2026-10-31 (see the local review delivery); a corrected/control training run has not been performed. The optional `idea-historical` training image is
**not yet built**. Commands and evidence boundaries are in
[verifier resources](evaluations/issue/verifier/resources/README.md).
Component terms are in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Review and execution guide

Scope: **issue only; no solution evaluation or solution training cost applies**. Run package validation from the repository root:

```bash
python -m autorlbench validate tasks/reward/idea_false_positive_reward
```

Evidence entry points: `workspace/config/.hydra/config.yaml`, `workspace/logs/rollouts/`, `workspace/logs/idea_analysis/`, and the W&B history under `workspace/logs/wandb/`. Reviewer-only case audits are under `evaluations/issue/verifier/resources/`. The symptom, mechanism and counterfactual
comparison are explained above; raw logs remain the source of truth. Historical
comparisons establish the documented case, not a guarantee over new seeds.

The task README, task card, issue oracle, solution oracle/reference patches and
verifier resources are reviewer material. Agents receive the level-specific
`instruction.md` and staged `workspace/`; do not place the reviewer README or
oracle in the workspace. Issue instructions use the common diagnosis scaffold.
Solution instructions describe the fixed workload, deliverables and acceptance
contract without specifying a repair. Upstream contribution policies and unrelated
utilities are excluded; the hidden round-two workspace audit records removals.
Production modules and frozen evidence are retained for diagnosis and import
compatibility; this is not a file-by-file minimal dependency extraction.

### Resource budget

Issue diagnosis uses CPU and frozen evidence: **0 GPU-hours**. The optional historical training recipe is outside this task's acceptance scope.

### Offline review, 2026-11-01

This review runs package validation, evidence/hash checks and applicable CPU
checks only. It does not run a model agent, hosted judge, GPU training, prepare
downloads, or checkpoint deserialization. GPU figures above are historical
measurements or configured timeout bounds; they are not measurements from this
review. Fresh end-to-end solution correctness remains unverified this round.

The offline source audit also joined a split assignment in the optional PPO critic
(`idea_source/verl/workers/critic/dp_critic.py`) so the snapshot parses. This is
a packaging syntax repair, not the reward defect or evidence of a new training run.
The hidden workspace audit records its before/after hashes.
