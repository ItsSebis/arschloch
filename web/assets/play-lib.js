// Pure helpers of the play page: no DOM, no network, so they can be tested
// with `node --test` (web/tests/play-lib.test.mjs).

/** Cards of a hand grouped by rank, ascending; the card order is kept. */
export function groupHand(hand) {
  const groups = [];
  for (const card of hand) {
    const last = groups[groups.length - 1];
    if (last && last.rank === card.rank) {
      last.cards.push(card);
    } else {
      groups.push({ rank: card.rank, label: card.label.slice(0, -1), cards: [card] });
    }
  }
  return groups;
}

export function cardColor(card) {
  return card.suit <= 1 ? "red" : "black";
}

export function toggleSelection(selected, id) {
  return selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id];
}

const byId = (view) => new Map(view.hand.map((c) => [c.id, c]));

// Rank first, then suit (Diamonds < Hearts < Spades < Clubs). Equal means a
// double-deck duplicate, which only the server's duplicate rule can settle.
const strength = (card) => card.rank * 4 + card.suit;
const topOf = (cards) => cards.reduce((best, c) => (strength(c) > strength(best) ? c : best));

/**
 * The cards to select when a rank is chosen as a whole: everything of that
 * rank on a lead, otherwise the weakest subset of the table's size that
 * beats it. Empty when the rank cannot be played.
 */
export function suggestSelection(view, rank) {
  const entry = view.playable.find((p) => p.rank === rank);
  if (!entry) return [];
  if (!view.table) return [...entry.card_ids];
  const size = view.table.cards.length;
  const cards = byId(view);
  const tableTop = strength(topOf(view.table.cards));
  const ids = entry.card_ids;
  // Strictly higher first; an equal card (a double-deck duplicate) is left
  // to the server's duplicate rule and only suggested when nothing is higher.
  for (const accept of [(a, b) => a > b, (a, b) => a >= b]) {
    for (let start = 0; start + size <= ids.length; start += 1) {
      const subset = ids.slice(start, start + size);
      if (accept(strength(topOf(subset.map((id) => cards.get(id)))), tableTop)) return subset;
    }
  }
  return [];
}

/** Whether `selected` (card ids) may be submitted now, and if not, why. */
export function selectionState(view, selected) {
  if (view.phase === "exchange") {
    const need = view.give_count;
    return selected.length === need
      ? { valid: true, reason: "" }
      : { valid: false, reason: `Choose ${need} card${need === 1 ? "" : "s"} to give (${selected.length} selected)` };
  }
  if (view.phase !== "playing" || view.to_move !== view.human_seat) {
    return { valid: false, reason: "Wait for your turn" };
  }
  if (selected.length === 0) return { valid: false, reason: "Select the cards you want to play" };
  const cards = byId(view);
  const chosen = selected.map((id) => cards.get(id));
  if (chosen.some((c) => !c)) return { valid: false, reason: "One of those cards is not in your hand" };
  if (new Set(chosen.map((c) => c.rank)).size > 1) {
    return { valid: false, reason: "The cards must all be the same rank" };
  }
  const entry = view.playable.find((p) => p.rank === chosen[0].rank);
  if (view.table) {
    const size = view.table.cards.length;
    if (!entry) return { valid: false, reason: "These cards cannot beat the cards on the table" };
    if (chosen.length !== size) {
      return { valid: false, reason: `Play exactly ${size} card${size === 1 ? "" : "s"} to follow` };
    }
    if (strength(topOf(chosen)) < strength(topOf(view.table.cards))) {
      return { valid: false, reason: "These cards do not beat the cards on the table" };
    }
  } else if (!entry) {
    return { valid: false, reason: "You cannot play those cards" };
  }
  return { valid: true, reason: "" };
}

// ------------------------------------------------------------ replaying events

/** The state the table shows while events are replayed one by one. */
export function initialDisplay(view) {
  return {
    handSizes: view.seats.map((s) => s.hand_size),
    places: view.seats.map((s) => s.place ?? null),
    table: view.table ?? null,
    toMove: view.to_move ?? null,
    leader: null,
    actions: view.seats.map(() => ""),
    passRule: view.pass_rule ?? "final",
    // Out of the current trick after passing (pass rule "final").
    passed: view.seats.map((s) => Boolean(s.passed)),
  };
}

const labels = (cards) => cards.map((c) => c.label).join(" ");

/** The display after `event`; the input is never modified. */
export function applyEvent(display, event) {
  const next = {
    ...display,
    handSizes: [...display.handSizes],
    places: [...display.places],
    actions: [...display.actions],
    passed: [...display.passed],
  };
  switch (event.type) {
    case "round_start":
      next.handSizes = [...event.hand_sizes];
      next.places = event.hand_sizes.map(() => null);
      next.actions = event.hand_sizes.map(() => "");
      next.passed = event.hand_sizes.map(() => false);
      next.table = null;
      next.leader = event.leader;
      break;
    case "play":
      next.handSizes[event.seat] = event.hand_left;
      next.table = { seat: event.seat, cards: event.cards };
      next.actions[event.seat] = `plays ${labels(event.cards)}`;
      break;
    case "pass":
      next.actions[event.seat] = "passes";
      if (display.passRule === "final") next.passed[event.seat] = true;
      break;
    case "trick_end":
      next.table = null;
      next.leader = event.leader;
      next.actions = next.actions.map(() => "");
      next.passed = next.passed.map(() => false);
      break;
    case "finished":
      next.places[event.seat] = event.place;
      break;
    default:
      break;
  }
  return next;
}

function ordinal(n) {
  const suffix = { 1: "st", 2: "nd", 3: "rd" }[n] ?? "th";
  return `${n}${suffix}`;
}

/** One log line for an event, or "" when it needs none. */
export function describeEvent(event, names, humanSeat) {
  const who = (seat) => names[seat];
  const verb = (seat, plural, singular) => (seat === humanSeat ? plural : singular);
  switch (event.type) {
    case "round_start":
      return `Round ${event.round} begins; ${who(event.leader)} ${verb(event.leader, "lead", "leads")}`;
    case "exchange":
      return event.pairs
        .map((p) => `${who(p.from)} ${verb(p.from, "give", "gives")} ${p.count} card${p.count === 1 ? "" : "s"} to ${who(p.to)}`)
        .join("; ");
    case "exchange_yours":
      return `You gave ${labels(event.gave)} and received ${labels(event.received)}`;
    case "play":
      return `${who(event.seat)} ${verb(event.seat, "play", "plays")} ${labels(event.cards)}`;
    case "pass":
      return `${who(event.seat)} ${verb(event.seat, "pass", "passes")}`;
    case "trick_end":
      return `${who(event.leader)} ${verb(event.leader, "lead", "leads")} the next trick`;
    case "finished":
      return `${who(event.seat)} ${verb(event.seat, "are", "is")} out in ${ordinal(event.place)} place`;
    case "round_end":
      return "Round over";
    case "match_end":
      return "Match over";
    default:
      return "";
  }
}

/** Milliseconds to show an event for at `speed` (0 = instant, 1 = normal). */
export function eventDelay(event, speed) {
  if (!speed) return 0;
  const base = { play: 900, pass: 450, finished: 600, trick_end: 700, round_start: 600, round_end: 900 }[event.type] ?? 0;
  return Math.round(base / speed);
}

// ------------------------------------------------------------------ records

export function formatScore(score) {
  if (!Number.isFinite(score)) return "–";
  const text = score.toFixed(2);
  return score > 0 ? `+${text}` : text.replace("-0.00", "0.00");
}

/** Maps a -1..+1 score to 0..1 for a bar. */
export function barFraction(score) {
  return Math.min(1, Math.max(0, (score + 1) / 2));
}
