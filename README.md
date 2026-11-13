# Mini Campus Delivery Robot

AI Fundamentals final project. A small console simulation integrates A* search, Bayesian inference and a one-step Markov decision process to help a delivery robot plan a campus route and decide whether to enter an uncertain corridor.

## Run

Requires Python 3.8 or newer. No packages, internet, API keys or training data are required. Extract the ZIP and open a terminal in the extracted project directory.

```bash
python main.py
python main.py --scenario alert --trace
python main.py --scenario clear
python main.py --scenario alert --prior 0.1
python -m unittest discover -v
```

On Windows, use `py` if `python` is unavailable. On macOS/Linux, use `python3` if needed. In Google Colab, upload `main.py` using the Files panel and run `!python main.py --trace` in a code cell. To run tests there, also upload `test_project.py` and run `!python -m unittest discover -v`.

## Problem and shared workflow

A delivery robot needs a shortest route around known walls, but a temporary obstacle may block its next cell. The map cannot fully describe this uncertainty. A* returns the route; the route identifies the next cell to observe; Bayes updates its blockage probability; the decision model uses that probability to choose Move or Wait. The simulation then advances one planned cell or keeps the robot at its current location.

This is a one-decision educational prototype. It does not simulate a complete delivery, collision outcomes or repeated sensor observations.

## Inputs and assumptions

- Grid: 5 by 5; coordinates are zero-based (row, column).
- Start: (0, 0); destination: (4, 4).
- Known static walls: (1, 1), (2, 1).
- Movement: four directions, unit cost, no diagonal steps.
- The sensor observes an UNKNOWN temporary obstacle at the next route cell; known walls remain excluded from the path.
- Prior P(blocked) = 0.30.
- P(alert | blocked) = 0.85; P(alert | clear) = 0.15.
- Negative readings use the complementary likelihoods: 0.15 and 0.85.
- All map data, sensor probabilities, rewards and transitions are hypothetical, predefined assumptions. They were not measured or learned.

## Algorithm 1 — A* search

Input: map, start and goal. A priority queue expands the node with the smallest f(n) = g(n) + h(n). Here g is the route cost so far and h is Manhattan distance to the goal. Better paths update the cost and parent dictionaries. Parent links reconstruct the final route.

Manhattan distance is admissible and consistent on this four-direction unit-cost map, so the returned path is shortest. Output includes the path, length and optional expansion trace. The default path has eight moves, matching the start-to-goal Manhattan lower bound. Known walls remain excluded; a second test map forces a different route.

Why selected: it combines known route cost with a goal-directed estimate. On larger maps a useful heuristic can reduce exploration; we do not claim a measured speed advantage on this tiny map.

## Algorithm 2 — Bayesian inference

Input: prior and one sensor reading. For observation O:

P(blocked | O) = P(O | blocked) P(blocked) / [P(O | blocked) P(blocked) + P(O | clear) P(clear)].

For alert: 0.85 * 0.30 / (0.85 * 0.30 + 0.15 * 0.70) = 0.708333.

For clear: 0.15 * 0.30 / (0.15 * 0.30 + 0.85 * 0.70) = 0.0703125.

Why selected: a sensor is imperfect; its likelihood is not the same as the posterior probability. Bayes combines prior information and evidence.

## Algorithm 3 — One-step MDP decision

States S = {clear, blocked}; actions A = {Move, Wait}; discount gamma = 0.9. Transition probabilities T(s' | s, a):

| Action | Current state | Next clear | Next blocked | Immediate reward |
|---|---|---:|---:|---:|
| Move | clear | 0.90 | 0.10 | 10 |
| Move | blocked | 0.20 | 0.80 | -20 |
| Wait | clear | 0.95 | 0.05 | -2 |
| Wait | blocked | 0.60 | 0.40 | -1 |

State means corridor condition at a decision epoch. Move models attempting entry, not guaranteed successful passage. Rewards depend on the current state: progress is rewarded, a blocked entry attempt is penalized and waiting costs time. Transitions model the corridor condition at the next epoch; the current demo executes only one decision.

The implementation computes one Bellman backup:

Q(s,a) = R(s,a) + gamma * sum over s' [T(s' | s,a) V0(s')].

The default terminal initialization is V0(clear) = V0(blocked) = 0. Consequently this first backup reduces to immediate reward. This is the one-step simplification explicitly permitted in the detailed assignment; transitions do not affect the default first-step recommendation. The code implements the transition term and a test verifies it with nonzero future values. This is not converged value iteration or a learned long-term policy.

The real state is uncertain. We use the Bayesian belief b(blocked)=p and b(clear)=1-p to score each action:

Score(a) = sum over s [b(s) Q(s,a)].

Therefore Score(Move) = 10 - 30p; Score(Wait) = -2 + p. Choose Move if p < 12/31 (about 0.3871); otherwise Wait, including ties. This is a belief-weighted one-step MDP action evaluation, not a full POMDP solver.

Why selected: it expresses the trade-off between progress, waiting and risky movement as explicit rewards.

## Results and evaluation

| Observation | Posterior blocked | Move score | Wait score | Recommendation |
|---|---:|---:|---:|---|
| Alert | 0.708333 | -11.250000 | -1.291667 | Wait |
| Clear | 0.0703125 | 7.890625 | -1.9296875 | Move |

Both scenarios use the same eight-move planned route. Alert keeps the robot at (0,0); Clear commands the first step to (0,1). Results are deterministic calculations under toy assumptions, not measured real-world accuracy or collision rates. Full console output with intermediate steps is in `sample_output.txt`.

Six automated tests pass: shortest and valid routes compared with an independent BFS reference on two maps; unreachable goal; start equals goal; both Bayesian posteriors and action choices; normalized transitions and a nonzero Bellman continuation; invalid probabilities. The test file groups related assertions into six test methods.

## Files

- `main.py`: application and all three algorithms implemented directly.
- `test_project.py`: correctness and integration tests.
- `sample_output.txt`: actual captured output of `python main.py --trace`.
- `DEFENSE_RU.md`: Russian explanations, English presentation script and Q&A.
- `README.md`: project documentation.

## Limitations and future development

The sensor model, rewards and state transitions need real calibration before practical use. The action model describes a local corridor rather than the full robot position and delivery state. Extend it with repeated observations, collision/outcome simulation, replanning, a larger state space and multi-step value iteration. A full model of hidden states would require belief updates over time and potentially a POMDP method. A GUI and real campus map are optional future extensions.

## AI assistance and attribution

This prototype and accompanying documentation were prepared with ChatGPT assistance. No external dataset, pretrained model or third-party implementation is included. The student should review and understand the code, adapt it as needed and disclose AI assistance according to course policy. Do not claim independent authorship of AI-assisted code. The project scope follows the supplied Final Project and MiniAI_FPTopics instructions, topic 02.
