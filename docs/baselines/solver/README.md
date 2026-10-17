# Endgame solver: measurements (Phase 21)

Binary: release build of the commit "solver: bound the search memo". All runs single-threaded per run on the owner's
machine; `cli solver-gap`, all seats `adaptive`, 200 single-round games, seed 1, budget 2,000,000 nodes per decision.
"Solved" is the share of decisions with more than one legal move that the solver handled (at most `T` cards unseen).

## Cost per hand-off threshold `T` (`max-unseen`)

| Deck | Players | T | Solved | Over budget | Nodes per solved decision (mean / max) | Time per 200 games |
|---|---|---|---|---|---|---|
| single | 3 | 4 | 7.5% | 0 | 1,389 / 79,250 | 1.0 s |
| single | 4 | 4 | 7.6% | 0 | 1,229 / 214,611 | 0.9 s |
| single | 5 | 4 | 7.1% | 0 | 1,186 / 78,558 | 0.8 s |
| single | 3 | 6 | 12.1% | 11 of 1147 | 14,867 / 1,272,287 | 34 s |
| single | 4 | 6 | 11.5% | 16 of 1076 | 14,522 / 570,414 | 42 s |
| single | 5 | 6 | 10.8% | 11 of 988 | 16,299 / 756,340 | 37 s |
| single | 3 | 8 | 15.1% | 172 of 1427 | 36,989 / 1,272,287 | 308 s |
| single | 4 | 8 | 14.0% | 149 of 1303 | 43,723 / 1,853,570 | 283 s |
| double | 4 | 4 | 3.8% | 1 of 734 | 3,316 / 424,535 | 4.0 s |
| double | 5 | 4 | 3.5% | 0 | 2,645 / 302,983 | 1.8 s |
| double | 6 | 4 | 3.4% | 0 | 1,829 / 166,111 | 1.2 s |
| double | 4 | 6 | 5.4% | 25 of 1050 | 14,822 / 465,870 | 54 s |
| double | 5 | 6 | 5.1% | 23 of 973 | 17,323 / 1,366,022 | 53 s |
| double | 6 | 6 | 5.1% | 16 of 949 | 19,185 / 1,845,309 | 43 s |

(The 5-player T=8 run and everything above T=8 were not measured: T=8 already goes over budget on about 12% of the
solved decisions and takes about five minutes per 200 games, so larger thresholds cannot work with this search.)

**Default: `max-unseen=6`, `budget=2000000`.** Criterion: the largest T with at most about 2% of solved decisions
over budget and under a minute per 200 single-round games. T=4 is nearly free but solves half as many decisions; T=8
is not usable. Over-budget decisions fall back to the fallback strategy (the match still completes).

**The memo matters.** The same 3-player T=4 run without the position memo (`--budget 20000000`): 40.5 s, mean 132,203
and max 18,526,659 nodes per decision, 3 over budget; with it: 1.0 s, 1,389 and 79,250, none over budget (about 95x
fewer nodes). The memo is bounded (500,000 positions, about 75 MB), because an unbounded memo ran the machine out of
memory in a measurement with a 20M-node budget.

## Head to head, luck-free (`--skill-score duplicate`)

4 players, single deck, 1200 matches x 5 rounds (300 duplicate groups), seed 7; the solver sits in every seat once
per deal. Score: +1 best .. -1 worst, +-SE.

| Table | Strategy | Skill score (duplicate) |
|---|---|---|
| solver + 3x `adaptive` | `EndgameSolver(adaptive,T=6)` | **+0.105 +-0.006** |
| | `adaptive` (each) | -0.035 +-0.002 |
| solver + `neat-v2` champion + `adaptive` + `lowest-legal` | `Neat(champion)` | **+0.628 +-0.008** |
| | `EndgameSolver(adaptive,T=6)` | -0.076 +-0.011 |
| | `adaptive` | -0.234 +-0.011 |
| | `lowest-legal` | -0.318 +-0.012 |

Note (stale number): an earlier measurement of this player and seed gave +0.105; the re-run in
`docs/baselines/solver-toggle/README.md` (`adaptive+solver:6`, same table, same seed 7) gives +0.100 +-0.006. The two
are 0.8 SE apart, i.e. within noise, and not a regression. Probably the games differ: the `endgame-solver` strategy returns forced moves itself while the
`+solver` wrapper hands them to `adaptive`, which draws random numbers for them, so the same seed does not replay the
same games (not re-run to confirm). The +0.105 above is the number of the old strategy and stays as measured.

Reading: the solver is clearly better than the strategy it falls back to (+0.14 in the first table, +0.16 in the
second; many standard errors), but the trained champion is far stronger than both: the solver plays exactly only about
11% of the decisions (the last few cards of a round), and the champion's strength is in the rest. Closing that gap
needs the early-game sampler (the spec's phase B), not a larger T.

## Where the strategies deviate (`cli solver-gap`)

4 players, single deck, 1500 games, seed 3, T=6: 68,538 decisions with a choice, 7,825 solved (11.4%), 170 over
budget. Share of the solved decisions where the strategy's move is worse in expectation than the solver's, and the
places lost per solved decision:

| Strategy | Solved | Differ | Share | Places lost each |
|---|---|---|---|---|
| `adaptive` | 2019 | 217 | 10.7% | 0.067 |
| `endgame-denial` | 1897 | 197 | 10.4% | 0.066 |
| `neat-v2` champion | 1878 | 214 | 11.4% | 0.077 |
| `lowest-legal` | 2031 | 309 | 15.2% | 0.127 |

What the costliest disagreements have in common (each costs 1.5-2.3 expected places; the explanations are my reading of the positions, the moves and costs are measured):

- **Leading a high single with several cards in hand** (`adaptive`: hand 10S 4H 3C, plays 10S; the solver leads 3C).
  Plausibly the high card is worth more kept as an answer to someone else's lead than spent on a lead.
- **Splitting a set** (`adaptive`: hand 4C 5H 5D 5C, plays 5D 5H; the solver plays all three fives). Leaving a single of a
  rank behind may leave a weak card to get rid of later; with few cards unseen the solver plays the whole group.
- **The same for `lowest-legal`** (hand 5C 5S: plays 5S, the solver plays the pair).
- **The champion and `endgame-denial` also lead a low single while holding a pair or triple that wins** (champion: 3D
  from 5H 3D 6C 5D 6H 5C where the three fives are the strong lead), a sign that none of them reads the end of the
  round exactly.

`cli solve` analyses any one of these positions move by move.

## Reproduce

`cli solver-gap --player-count 4 --games 200 --seed 1 --max-unseen 6 --strategy adaptive ...` (one `--strategy` per
seat); the head to head is `cli --player-count 4 --matches 1200 --rounds 5 --seed 7 --skill-score duplicate --strategy
endgame-solver --strategy adaptive --strategy adaptive --strategy adaptive` and the same with
`--strategy neat:docs/baselines/neat-v2/champion.json --strategy adaptive --strategy lowest-legal` as seats 2-4.
