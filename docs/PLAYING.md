# Playing against the models

`cli play` serves a small game page where you sit at a table and play full
Arschloch matches against trained models and the hand-written strategies. The
rules are the ones in `docs/RULES.md`; the other seats are played by the same
`Strategy` implementations the simulator uses.

## Start

```bash
cargo build --release -p cli
target/release/cli play
```

Open the printed address (`http://127.0.0.1:8090/`). The page listens on your
own machine only. A trained champion (`Neat(champion-v2)`, the committed
baseline) is built in, so this works from a release download too. To play
against your own models, add them:

```bash
target/release/cli play --model runs/first --model runs/second/gen-0042.json
```

`--model` takes a genome file written by `cli train` or a run directory (its
`best.json`). Other options: `--port N` (default 8090) and `--records FILE`
(default `play-records.jsonl` in the current directory).

## Setting up a game

Choose the table size (3-6), the deck (single or double, with the duplicate
rule for the double deck), whether a pass ends your part in the trick (the
rules, default) or you may still play later in it (the old simulator
behaviour), how many rounds make a match (default 8), your seat
(or random) and, for every other seat, an opponent: a trained model or a
hand-written strategy. A match runs for the chosen number of rounds with the
roles carried over, exactly like a simulated match.

## Playing

- **Your hand** is grouped by rank. Click cards to select them (click again to
  deselect); click a rank's label to select it as a whole: on a lead the whole
  rank, otherwise the weakest cards that beat the table. Cards that cannot be
  played now are dimmed.
- **Play selected** is enabled only when the selection is legal; the line above
  your hand says why it is not (wrong number of cards, not higher, mixed ranks).
  Enter plays, Esc clears. You may play any cards of one rank that beat the
  table, not just the weakest or strongest.
- **Pass** is allowed when you do not lead the trick. Under the default rule a
  pass is final for the trick: you sit out until it ends (the seat boxes show
  who is out of the trick).
- **The exchange**: if you hold a lower role you choose which cards to give
  (as many as the role asks for); a higher role automatically hands back its
  lowest cards. The log tells you what you gave and received.
- **Replay speed** controls how fast the other seats' moves are shown. The game
  itself runs on the server, so reloading the page and pressing *Continue last
  game* picks it up again.

## Ask a model

*Ask a model* shows how a trained model ranks your legal moves in the current
position, with its score for each (higher is better in its eyes; the top one is
what it would play). The model sees only what you see. Click a suggestion to
select those cards.

## Your record

Finished matches are appended to the records file, one JSON line each. *Your
record* shows your mean role score (+1 = always President, 0 = even, -1 = always
last) per opponent and per table, how often you finished first and last, and the
latest games. For reference, the built-in champion scores about +0.6 against the
hand-written strategies (`docs/baselines/neat-v2`); the same scale applies to
you.

## Security

The server answers requests addressed to localhost only and refuses POST
requests that come from other web pages (checked by `Origin`), so another site
cannot play or write records through your browser. It has no accounts: anyone
who can reach the port on your machine can play.
