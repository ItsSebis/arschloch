# Arschloch — Rules Reference

This is the authoritative rules document for the simulator. It merges the
German Wikipedia article on *Arschloch (Kartenspiel)* with house-rule
decisions made explicitly for this project (recorded inline as
**Decision:**). Anything not covered here is out of scope until a future
roadmap phase (see `ROADMAP.md`).

Source: https://de.wikipedia.org/wiki/Arschloch_(Kartenspiel)

## Players & Deck

- Supported table sizes: 3, 4, 5, 6 players.
- Deck variant is chosen per match:
  - **Single deck**: standard 52-card poker deck (ranks 2–10, J, Q, K, A;
    suits Clubs, Spades, Hearts, Diamonds).
  - **Double deck**: two shuffled-together 52-card decks (104 cards), used
    for larger groups. This introduces true duplicate cards (same rank
    *and* suit appearing twice) — see "Duplicate cards" below.
- **Decision:** cards are dealt as evenly as possible; with a remainder,
  the excess cards go to the earliest players in deal order (an
  implementation detail — `sim`'s batch runner rotates which strategy
  occupies which seat across a batch specifically so this doesn't bias
  aggregate strategy comparisons; a single match's finishing order can
  still be affected by seat position).

## Card Ranking

- Rank order (low → high): `2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A`.
- **Decision — Suit ranking (non-standard for this game, house rule):**
  suits break ties between cards of otherwise-equal rank. Suit order
  (low → high): `Diamonds < Hearts < Spades < Clubs`.
  - i.e. Hearts beats Diamonds, Spades beats Hearts, Clubs beats Spades.
  - This matters because with 4 suits per rank (single deck) or 8 per rank
    (double deck), a play of "the same numeric rank" needs a tiebreaker to
    produce a strict total order over cards.

### Duplicate cards (double-deck variant only)

In the double-deck variant, the exact same card (rank *and* suit) can
appear twice. Suit ranking cannot break this tie, so a second, configurable
rule applies, chosen per match:

- `FirstDealtWins` — of two identical cards, the copy dealt earlier in the
  shuffle is considered higher-ranked.
- `LastDealtWins` — the copy dealt later is considered higher-ranked.

This is tracked internally by tagging each card with a deal-order index at
shuffle time; the index is only ever consulted when rank and suit are both
equal. In the single-deck variant this rule is never exercised.

## Roles

Roles are assigned at the end of each round based on the order in which
players emptied their hand (1st out → highest role, last player left →
lowest role / Arschloch). Role names depend on table size:

| Players | Roles (highest → lowest) |
|---|---|
| 3 | President, Dorftrottel, Arschloch |
| 4 | President, Vize, Vize-Arschloch, Arschloch |
| 5 | President, Vize, Dorftrottel, Vize-Arschloch, Arschloch |
| 6 | President, Vize, Offizier, Dummkopf, Vize-Arschloch, Arschloch |

In the first game of a session there are no prior roles, so the first
round is played with roles undefined; the finishing order of that round
establishes roles for the exchange before round 2.

## Card Exchange ("Drücken")

Before every round after the first, roles exchange cards. The
higher-ranked role hands back their **lowest** N cards, chosen naively
(no strategic input on that side). The lower-ranked role, however, hands
over any N cards of its own choosing: which cards it gives up is decided
by its `Strategy` (see `docs/ARCHITECTURE.md` for which strategies do
what — e.g. `HoldBackPairs` will keep a pair of Aces intact and instead
give up an isolated 9 and Jack). The exchange itself is still mandatory
— there is no discretion about *whether* to exchange, only about
*which* specific cards the lower-ranked role gives up, which is now
strategy-driven.

This was originally planned as merely a tie-break among the "highest N"
cards on the giving side, but that framing turned out not to make sense:
under `Card::compare`'s total order (rank, then suit, then a
deal-index tiebreak — see "Card Ranking" above), no two cards in a hand
are ever truly equal, so a literal "break ties among the Nth-highest
cards" rule would have nothing to ever act on. The real design goal —
letting a strategy avoid splitting up a useful same-rank reserve when
it's forced to exchange — required giving strategies full freedom over
which cards to give up, not just a tie-break among equally-ranked
candidates.

**Decision — exchange counts scale with table size** (outer role-pair
exchanges the most, decreasing by 1 per pair moving inward, a lone
unpaired middle role exchanges nothing):

| Players | Exchange pairs (count) |
|---|---|
| 3 | President ↔ Arschloch: **1**; Dorftrottel: 0 |
| 4 | President ↔ Arschloch: **2**; Vize ↔ Vize-Arschloch: **1** |
| 5 | President ↔ Arschloch: **2**; Vize ↔ Vize-Arschloch: **1**; Dorftrottel: 0 |
| 6 | President ↔ Arschloch: **3**; Vize ↔ Vize-Arschloch: **2**; Offizier ↔ Dummkopf: **1** |

> The 3-player row was inferred by extrapolating the pattern from the
> 4/5/6-player answers (outer pair count = number of role-pairs at that
> table size); it was not given explicitly and should be double-checked
> against real play before trusting simulator output for 3-player games.

## Playing a Round

1. **Lead.** The active leader plays a **combo**: one or more cards of
   equal rank (a single, pair, triple, …). Phase 0/1 scope is singles and
   same-rank sets only — no straights, no bombs/revolution (see
   "Deferred rules" below).
2. **Follow.** Going around the table, each other player either:
   - **Passes** (always legal, even if they hold a card/combo that could
     beat the current play — this is an intentional strategic option the
     simulator needs to model, not just a fallback when unable to beat), or
   - Plays a combo of the **same size**, with **strictly higher** rank
     (suit, then the duplicate-tiebreak rule, break ties within equal
     rank).
   - **Decision:** when a combo has more than one card, its **highest**
     card (by the same rank/suit/duplicate-tiebreak order used everywhere
     else) represents the whole combo when comparing it against another
     combo of the same size.
3. **Trick ends** when every other active player has passed in sequence.
   The last player to play a combo wins the trick, collects nothing (cards
   are discarded, not collected — unlike Whist-style games), and leads the
   next trick.
4. A player who empties their hand is removed from further trick-leading
   for the round and recorded in finishing order. Passing is not available
   to a player with no cards — they're simply skipped.
5. The round ends when only one player still holds cards; that player is
   ranked last (Arschloch, or the table's lowest role).

### First lead of a trick / round

- **First round of the first game in a session:** the player holding the
  single lowest card (by rank, then suit, then duplicate-tiebreak) leads
  the first trick. They may open with any legal combo (not required to
  include that specific card).
- **Every later round:** the player who was Arschloch (lowest role) in the
  previous round leads the first trick of the new round.
- **Every trick after the first in a round:** the winner of the previous
  trick leads.

## Deferred / Out-of-scope Rules

These appear in variants of the real game or the Wikipedia article but are
explicitly **not** implemented in the current phases. They're recorded
here so a future roadmap phase can pick them up deliberately rather than
re-researching from scratch:

- **Bombs / Revolution** — playing four-of-a-kind to auto-win a trick or
  invert rank order for the rest of the round.
- **Straights** — sequences of consecutive ranks as a playable combo.
- **Jokers** — not used; both deck variants are Joker-free.
- **Rule-violation penalty** — some house rules auto-demote a player to
  Arschloch for an illegal play; the simulator only ever generates legal
  moves, so this doesn't apply to simulated strategies, but would matter
  if a human/UI player is ever added.
- **"Ace as last card" auto-Arschloch** — a superstition/house-rule variant
  where playing an Ace as your very last card makes you Arschloch
  regardless of finishing order.
