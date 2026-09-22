#!/usr/bin/env python3
"""Play two UCI engines against each other using python-chess.

Usage:
  python3 scripts/uci_match.py --engine1 "./target/release/chess-engine --uci" --engine2 "$(which stockfish)" \
      --games 6 --move-time 200 --out match.pgn
  python3 scripts/uci_match.py --engine1 ./target/release/chess-engine --engine2 "$(which stockfish)" \
      --games 6 --nodes "2000:1000"
  python3 scripts/uci_match.py --engine1 A --engine2 B --games 6 --depth "6:5"
"""
import argparse
import re
import subprocess
import sys
import time

import chess
import chess.pgn


class GoTimeout(Exception):
    pass


class UciEngine:
    def __init__(self, cmd: str, name: str):
        self.name = name
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.lines = iter(self.proc.stdout)
        self.go_timeout = 120.0  # wall-clock cap per search (default)
        self.send("uci")
        self.expect("uciok")
        self.send("isready")
        self.expect("readyok")

    def send(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def next_line(self):
        while True:
            line = next(self.lines).strip()
            if line:
                return line

    def expect(self, token: str, timeout: float = 30.0):
        deadline = time.time() + timeout
        while True:
            line = self.next_line()
            if token in line:
                return line
            if time.time() > deadline:
                raise RuntimeError(f"{self.name}: timed out waiting for {token}")

    def set_position(self, moves):
        if moves:
            self.send("position startpos moves " + " ".join(moves))
        else:
            self.send("position startpos")

    def go(self, go_cmd: str):
        self.send("go " + go_cmd)
        deadline = time.time() + self.go_timeout
        while True:
            line = self.next_line()
            if line.startswith("bestmove"):
                m = re.match(r"bestmove\s+(\S+)", line)
                return m.group(1) if m else None
            if time.time() > deadline:
                raise GoTimeout(f"{self.name}: no bestmove within {self.go_timeout}s")

    def quit(self):
        try:
            self.send("quit")
        except BrokenPipeError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def parse_limit(spec: str, default: str):
    a, _, b = spec.partition(":")
    return (a or default, b or default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine1", required=True, help="cmd for first engine")
    ap.add_argument("--engine2", required=True, help="cmd for second engine")
    ap.add_argument("--name1", default="engine1")
    ap.add_argument("--name2", default="engine2")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--move-time", type=int, default=None, help="ms per move (both)")
    ap.add_argument("--nodes", default=None, help="nodes per move 'n1:n2'")
    ap.add_argument("--depth", default=None, help="depth per move 'd1:d2'")
    ap.add_argument("--out", default=None, help="PGN output file")
    ap.add_argument("--go-timeout", type=float, default=120.0)
    args = ap.parse_args()

    go1 = None
    go2 = None
    if args.move_time:
        go1 = go2 = "movetime %d" % args.move_time
    elif args.nodes:
        n1, n2 = parse_limit(args.nodes, "1000")
        go1, go2 = "nodes %s" % n1, "nodes %s" % n2
    elif args.depth:
        d1, d2 = parse_limit(args.depth, "6")
        go1, go2 = "depth %s" % d1, "depth %s" % d2
    else:
        ap.error("provide one of --move-time, --nodes, --depth")

    e1 = UciEngine(args.engine1, args.name1)
    e1.go_timeout = args.go_timeout
    e2 = UciEngine(args.engine2, args.name2)
    e2.go_timeout = args.go_timeout

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    aborted = 0
    games = []

    for game_no in range(1, args.games + 1):
        white = e1 if game_no % 2 == 1 else e2
        black = e2 if white is e1 else e1
        go_white = go1 if white is e1 else go2
        go_black = go2 if black is e2 else go1

        moves = []
        result = None
        timed = False
        for ply in range(400):
            mover = white if (ply % 2 == 0) else black
            go_cmd = go_white if mover is white else go_black
            mover.set_position(moves)

            board = chess.Board()
            for m in moves:
                board.push_uci(m)
            legal = {m.uci() for m in board.legal_moves}

            uci_move = mover.go(go_cmd)
            if uci_move and uci_move in legal:
                board.push_uci(uci_move)
                moves.append(uci_move)
            else:
                print(f"  GAME {game_no} ABORTED: {mover.name} played illegal {uci_move!r} "
                      f"after {' '.join(moves) or 'startpos'}")
                aborted += 1
                result = "aborted"
                break

            if board.is_checkmate():
                result = "1-0" if board.turn == chess.BLACK else "0-1"
                break
            if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                result = "1/2-1/2"
                break

        if result == "aborted":
            continue
        if result is None:
            result = "1/2-1/2"
        results[result] += 1

        game = chess.pgn.Game()
        board = chess.Board()
        node = game
        for u in moves:
            node = node.add_variation(chess.Move.from_uci(u))
        game.headers.update({
            "Event": "Engine Match",
            "Site": "cli",
            "Date": time.strftime("%Y.%m.%d"),
            "Round": str(game_no),
            "White": white.name,
            "Black": black.name,
            "Result": result,
        })
        games.append(game)
        print(f"  {game_no:3d}  {white.name:>14} vs {black.name:<14}  {result}  ({len(moves)} plies)")

    e1.quit()
    e2.quit()

    score1 = 0.0
    score2 = 0.0
    for g in games:
        res = g.headers["Result"]
        if res == "1-0":
            score1 += 1.0 if g.headers["White"] == args.name1 else 0.0
            score2 += 1.0 if g.headers["White"] == args.name2 else 0.0
        elif res == "0-1":
            score1 += 1.0 if g.headers["Black"] == args.name1 else 0.0
            score2 += 1.0 if g.headers["Black"] == args.name2 else 0.0
        else:
            score1 += 0.5
            score2 += 0.5

    print("\nFinal score:")
    print(f"  {args.name1}: {score1}")
    print(f"  {args.name2}: {score2}")
    print(f"  W/D/L: {results['1-0']}/{results['1/2-1/2']}/{results['0-1']}   aborted: {aborted}")

    if args.out:
        with open(args.out, "w") as f:
            for g in games:
                f.write(str(g))
                f.write("\n\n")
        print(f"PGN written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())