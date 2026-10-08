// Run with: node --test web/tests
import assert from "node:assert/strict";
import test from "node:test";
import {
  applyEvent, barFraction, cardColor, describeEvent, eventDelay, formatScore, groupHand,
  initialDisplay, selectionState, suggestSelection, toggleSelection,
} from "../assets/play-lib.js";

const card = (id, rank, suit) => ({ id, rank, suit, label: `${"23456789TJQKA"[rank]}${"dhsc"[suit]}` });

// hand (weakest first): 4d(1) 9h(2) 9s(3) 9c(4) Kd(5)
const hand = [card(1, 2, 0), card(2, 7, 1), card(3, 7, 2), card(4, 7, 3), card(5, 11, 0)];
const playable = (sizesByRank) => Object.entries(sizesByRank).map(([rank, [ids, sizes]]) => ({ rank: Number(rank), card_ids: ids, sizes }));

const leadView = {
  phase: "playing", human_seat: 0, to_move: 0, must_lead: true, give_count: 0, hand, table: null,
  playable: playable({ 2: [[1], [1]], 7: [[2, 3, 4], [1, 2, 3]], 11: [[5], [1]] }),
};
// A pair of 8s (rank 6) on the table, topped by 8h.
const followView = {
  ...leadView, must_lead: false,
  table: { seat: 1, cards: [card(20, 6, 0), card(21, 6, 1)] },
  playable: playable({ 7: [[2, 3, 4], [2]] }),
};

test("groupHand groups by rank in ascending order and keeps the card order", () => {
  const groups = groupHand(hand);
  assert.deepEqual(groups.map((g) => [g.label, g.cards.map((c) => c.id)]), [["4", [1]], ["9", [2, 3, 4]], ["K", [5]]]);
  assert.deepEqual(groupHand([]), []);
});

test("cardColor is red for diamonds and hearts only", () => {
  assert.deepEqual([0, 1, 2, 3].map((s) => cardColor({ suit: s })), ["red", "red", "black", "black"]);
});

test("toggleSelection adds, removes and never duplicates", () => {
  assert.deepEqual(toggleSelection([], 3), [3]);
  assert.deepEqual(toggleSelection([3], 5), [3, 5]);
  assert.deepEqual(toggleSelection([3, 5], 3), [5]);
});

test("selectionState on a lead accepts any one-rank selection and explains the rest", () => {
  assert.equal(selectionState(leadView, []).valid, false);
  assert.match(selectionState(leadView, []).reason, /select/i);
  assert.equal(selectionState(leadView, [2]).valid, true);
  assert.equal(selectionState(leadView, [2, 3]).valid, true);
  assert.equal(selectionState(leadView, [2, 3, 4]).valid, true);
  const mixed = selectionState(leadView, [1, 2]);
  assert.equal(mixed.valid, false);
  assert.match(mixed.reason, /same rank/);
  assert.equal(selectionState(leadView, [99]).valid, false);
});

test("selectionState when following needs the table's size and a higher card", () => {
  assert.equal(selectionState(followView, [2, 3]).valid, true); // 9h 9s beats 8h 8d
  assert.equal(selectionState(followView, [3, 4]).valid, true);
  const one = selectionState(followView, [2]);
  assert.equal(one.valid, false);
  assert.match(one.reason, /2 cards/);
  const three = selectionState(followView, [2, 3, 4]);
  assert.equal(three.valid, false);
  // A rank that cannot beat the table at all.
  assert.equal(selectionState(followView, [1]).valid, false);
  assert.match(selectionState(followView, [1]).reason, /beat/);
});

test("selectionState is not valid out of turn", () => {
  const waiting = { ...leadView, to_move: 2 };
  const state = selectionState(waiting, [2]);
  assert.equal(state.valid, false);
  assert.match(state.reason, /turn/);
});

test("the exchange needs exactly the number of cards asked for", () => {
  const exchange = { ...leadView, phase: "exchange", to_move: null, give_count: 2 };
  assert.equal(selectionState(exchange, [1]).valid, false);
  assert.match(selectionState(exchange, [1]).reason, /2/);
  assert.equal(selectionState(exchange, [1, 5]).valid, true);
  assert.equal(selectionState(exchange, [1, 5, 2]).valid, false);
});

test("suggestSelection takes the whole rank on a lead and the weakest beating subset when following", () => {
  assert.deepEqual(suggestSelection(leadView, 7), [2, 3, 4]);
  assert.deepEqual(suggestSelection(leadView, 2), [1]);
  assert.deepEqual(suggestSelection(followView, 7), [2, 3]);
  assert.deepEqual(suggestSelection(followView, 2), []); // not playable
});

test("suggestSelection skips subsets that do not beat the table by suit", () => {
  // Table: a single 9s. Group of 9s: 9d(10) 9h(11) 9s(12) 9c(13): only 9c beats 9s.
  const view = {
    ...leadView, must_lead: false,
    hand: [card(10, 7, 0), card(11, 7, 1), card(12, 7, 2), card(13, 7, 3)],
    table: { seat: 1, cards: [card(30, 7, 2)] },
    playable: playable({ 7: [[10, 11, 12, 13], [1]] }),
  };
  assert.deepEqual(suggestSelection(view, 7), [13]);
});

test("applyEvent follows a round event by event", () => {
  let display = initialDisplay({ seats: [{ hand_size: 3, place: null }, { hand_size: 3, place: null }, { hand_size: 3, place: null }], table: null, to_move: 0 });
  display = applyEvent(display, { type: "round_start", round: 2, roles: null, leader: 1, hand_sizes: [4, 4, 4] });
  assert.deepEqual(display.handSizes, [4, 4, 4]);
  display = applyEvent(display, { type: "play", seat: 1, cards: [card(9, 3, 0)], hand_left: 3 });
  assert.deepEqual(display.handSizes, [4, 3, 4]);
  assert.equal(display.table.seat, 1);
  assert.match(display.actions[1], /5/);
  display = applyEvent(display, { type: "pass", seat: 2 });
  assert.equal(display.actions[2], "passes");
  display = applyEvent(display, { type: "finished", seat: 1, place: 1 });
  assert.equal(display.places[1], 1);
  display = applyEvent(display, { type: "trick_end", leader: 0 });
  assert.equal(display.table, null);
  assert.equal(display.leader, 0);
});

test("applyEvent does not modify its input", () => {
  const before = initialDisplay({ seats: [{ hand_size: 3, place: null }, { hand_size: 3, place: null }, { hand_size: 3, place: null }], table: null, to_move: 0 });
  const snapshot = JSON.stringify(before);
  applyEvent(before, { type: "play", seat: 0, cards: [card(9, 3, 0)], hand_left: 2 });
  assert.equal(JSON.stringify(before), snapshot);
});

test("describeEvent reads naturally and marks the human", () => {
  const names = ["You", "Neat(champion)", "LowestLegal"];
  assert.equal(describeEvent({ type: "pass", seat: 1 }, names, 0), "Neat(champion) passes");
  assert.match(describeEvent({ type: "play", seat: 0, cards: [card(1, 2, 0), card(2, 2, 1)], hand_left: 5 }, names, 0), /You play 4d 4h/);
  assert.match(describeEvent({ type: "finished", seat: 2, place: 1 }, names, 0), /LowestLegal.*1st/);
  assert.match(describeEvent({ type: "exchange", pairs: [{ from: 2, to: 1, count: 2 }] }, names, 0), /LowestLegal gives 2 cards to Neat\(champion\)/);
  assert.match(describeEvent({ type: "exchange_yours", gave: [card(1, 2, 0)], received: [card(5, 11, 0)] }, names, 0), /gave 4d.*received Kd/);
  assert.equal(describeEvent({ type: "trick_end", leader: 1 }, names, 0), "Neat(champion) leads the next trick");
  assert.equal(describeEvent({ type: "match_end" }, names, 0), "Match over");
});

test("scores are formatted and mapped onto a bar", () => {
  assert.equal(formatScore(0.6234), "+0.62");
  assert.equal(formatScore(-1), "-1.00");
  assert.equal(formatScore(0), "0.00");
  assert.equal(barFraction(-1), 0);
  assert.equal(barFraction(1), 1);
  assert.equal(barFraction(0), 0.5);
  assert.equal(barFraction(5), 1);
});

test("the replay delay scales with the speed and a lead is slower than a pass", () => {
  assert.ok(eventDelay({ type: "play" }, 1) > eventDelay({ type: "pass" }, 1));
  assert.ok(eventDelay({ type: "play" }, 2) < eventDelay({ type: "play" }, 1));
  assert.equal(eventDelay({ type: "play" }, 0), 0); // instant
  assert.equal(eventDelay({ type: "exchange" }, 1), 0);
});
