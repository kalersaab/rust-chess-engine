use crate::board::chess_move::{ChessMove, MoveType};
use crate::board::pieces::piece_to_fen_char;
use crate::board::{Board, Color};
use crate::evaluation::Score;
use crate::search::Searcher;
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

pub struct EngineInfo {
    pub name: String,
    pub author: String,
}

struct GoParams {
    depth: Option<u32>,
    nodes: Option<u64>,
    movetime: Option<u128>,
    clock_ms: Option<u128>,
    infinite: bool,
    ponder: bool,
}

pub struct UciEngine {
    pub info: EngineInfo,
    pub board: Board,
    pub options: HashMap<String, String>,
    pub searcher: Searcher,
    stop_flag: Arc<AtomicBool>,
    search_thread: Option<JoinHandle<()>>,
    pondering: bool,
    ponder_tx: Option<mpsc::SyncSender<()>>,
}

impl UciEngine {
    pub fn new() -> Self {
        UciEngine {
            info: EngineInfo {
                name: "Rust Chess Engine".to_string(),
                author: "Unknown".to_string(),
            },
            board: Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .expect("Failed to create starting position"),
            options: HashMap::new(),
            searcher: Searcher::new(),
            stop_flag: Arc::new(AtomicBool::new(false)),
            search_thread: None,
            pondering: false,
            ponder_tx: None,
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        let reader = stdin.lock();

        for line in reader.lines() {
            if let Ok(input) = line {
                let trimmed = input.trim();
                if !trimmed.is_empty() {
                    self.handle_command(trimmed);
                }
            }
        }

        self.stop_and_join();
    }

    fn handle_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();

        if parts.is_empty() {
            return;
        }

        match parts[0] {
            "uci" => self.cmd_uci(),
            "isready" => self.cmd_isready(),
            "setoption" => self.cmd_setoption(&parts),
            "position" => {
                self.stop_and_join();
                self.cmd_position(&parts);
            }
            "go" => {
                self.stop_and_join();
                self.cmd_go(&parts);
            }
            "stop" => {
                self.stop_and_join();
            }
            "ponderhit" => {
                self.pondering = false;
                if let Some(tx) = self.ponder_tx.take() {
                    let _ = tx.send(());
                }
            }
            "ucinewgame" => {
                self.stop_and_join();
                self.searcher = Searcher::new();
                self.options.remove("Book");
            }
            "quit" => {
                self.stop_and_join();
                self.cmd_quit();
            }
            "display" => self.board.print(),
            "perft" => self.cmd_perft(&parts),
            _ => {}
        }
    }

    fn cmd_uci(&self) {
        println!("id name {}", self.info.name);
        println!("id author {}", self.info.author);
        println!("option name Hash type spin default 16 min 1 max 256");
        println!("option name Book type check default false");
        println!("option name Ponder type check default false");
        println!("option name GpuEnabled type check default false");
        println!("option name GpuBatchMin type spin default 8 min 2 max 64");
        println!("option name GpuMaxDepth type spin default 16 min 1 max 64");
        println!("uciok");
    }

    fn cmd_isready(&self) {
        println!("readyok");
    }

    fn cmd_setoption(&mut self, parts: &[&str]) {
        if parts.len() >= 5 && parts[1] == "name" && parts[3] == "value" {
            let name = parts[2];
            let value = parts[4..].join(" ");

            match name {
                "Hash" => {
                    if let Ok(size) = value.parse::<u32>() {
                        if size >= 1 && size <= 256 {
                            self.options.insert(name.to_string(), value.clone());
                            self.searcher.set_tt_size(size);
                        }
                    }
                }
                "Book" => {
                    self.options.insert(name.to_string(), value);
                }
                "GpuEnabled" => {
                    let enabled = value == "true";
                    self.options.insert(name.to_string(), value);
                    self.searcher.set_gpu_enabled(enabled);
                }
                "GpuBatchMin" => {
                    if let Ok(min) = value.parse::<usize>() {
                        self.options.insert(name.to_string(), value.clone());
                        self.searcher.set_gpu_batch_min(min);
                    }
                }
                "GpuMaxDepth" => {
                    if let Ok(depth) = value.parse::<u32>() {
                        self.options.insert(name.to_string(), value.clone());
                        self.searcher.set_gpu_max_depth(depth);
                    }
                }
                _ => {
                    self.options.insert(name.to_string(), value);
                }
            }
        }
    }

    pub fn get_option(&self, name: &str) -> Option<&String> {
        self.options.get(name)
    }

    fn cmd_position(&mut self, parts: &[&str]) {
        if parts.len() < 2 {
            return;
        }

        match parts[1] {
            "startpos" => {
                self.board =
                    Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                        .expect("Failed to load startpos");

                if parts.len() > 2 && parts[2] == "moves" {
                    self.apply_moves(&parts[3..]);
                }
            }
            "fen" => {
                if parts.len() >= 3 {
                    let mut fen_end = parts.len();
                    for (i, &part) in parts.iter().enumerate().skip(2) {
                        if part == "moves" {
                            fen_end = i;
                            break;
                        }
                    }

                    let fen = parts[2..fen_end].join(" ");

                    if let Ok(board) = Board::from_fen(&fen) {
                        self.board = board;

                        if fen_end < parts.len() && parts[fen_end] == "moves" && fen_end + 1 < parts.len() {
                            self.apply_moves(&parts[fen_end + 1..]);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn apply_moves(&mut self, moves: &[&str]) {
        for move_str in moves {
            if let Err(_) = self.apply_uci_move(move_str) {
                break;
            }
        }
    }

    fn apply_uci_move(&mut self, move_notation: &str) -> Result<(), String> {
        if move_notation.len() < 4 || move_notation.len() > 5 {
            return Err("Invalid move notation".to_string());
        }

        let from = &move_notation[0..2];
        let to = &move_notation[2..4];
        let from_sq = Board::parse_square_public(from)?;
        let to_sq = Board::parse_square_public(to)?;

        let promotion = move_notation.as_bytes().get(4).map(|&c| {
            if c.is_ascii_alphabetic() {
                (c as char).to_ascii_lowercase()
            } else {
                c as char
            }
        });

        let legal_move = self.board.generate_moves().into_iter().find(|m| {
            m.from == from_sq
                && m.to == to_sq
                && uci_promo_char(m) == promotion
        });

        match legal_move {
            Some(mv) => self.board.execute_move(mv.from, mv.to, mv.move_type),
            None => Err(format!("Illegal move: {}", move_notation)),
        }
    }

    pub fn current_fen(&self) -> String {
        self.board.to_fen()
    }

    pub fn get_legal_moves(&self) -> Vec<String> {
        self.board
            .generate_moves()
            .iter()
            .map(uci_move_string)
            .collect()
    }

    fn parse_go(&mut self, parts: &[&str]) -> GoParams {
        let mut params = GoParams {
            depth: None,
            nodes: None,
            movetime: None,
            clock_ms: None,
            infinite: false,
            ponder: false,
        };

        let mut wtime: Option<u128> = None;
        let mut btime: Option<u128> = None;
        let mut winc: Option<u128> = None;
        let mut binc: Option<u128> = None;

        let mut i = 1;
        while i < parts.len() {
            match parts[i] {
                "depth" if i + 1 < parts.len() => {
                    params.depth = parts[i + 1].parse::<u32>().ok().map(|d| d.min(64));
                    i += 2;
                }
                "nodes" if i + 1 < parts.len() => {
                    params.nodes = parts[i + 1].parse::<u64>().ok();
                    i += 2;
                }
                "movetime" if i + 1 < parts.len() => {
                    params.movetime = parts[i + 1].parse::<u128>().ok();
                    i += 2;
                }
                "wtime" if i + 1 < parts.len() => {
                    wtime = parts[i + 1].parse::<u128>().ok();
                    i += 2;
                }
                "btime" if i + 1 < parts.len() => {
                    btime = parts[i + 1].parse::<u128>().ok();
                    i += 2;
                }
                "winc" if i + 1 < parts.len() => {
                    winc = parts[i + 1].parse::<u128>().ok();
                    i += 2;
                }
                "binc" if i + 1 < parts.len() => {
                    binc = parts[i + 1].parse::<u128>().ok();
                    i += 2;
                }
                "infinite" => {
                    params.infinite = true;
                    i += 1;
                }
                "ponder" => {
                    params.ponder = true;
                    i += 1;
                }
                "searchmoves" => {
                    i += 1;
                    while i < parts.len() && is_uci_move_token(parts[i]) {
                        i += 1;
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }

        if params.movetime.is_none() {
            let (remaining, inc) = if self.board.turn == Color::White {
                (wtime, winc)
            } else {
                (btime, binc)
            };
            params.clock_ms = remaining.map(|r| {
                let inc_ms = inc.unwrap_or(0);
                let hard = r / 20 + inc_ms * 3 / 4;
                let cap = r * 9 / 10;
                hard.min(cap).max(50)
            });
        }

        params
    }

    fn cmd_go(&mut self, parts: &[&str]) {
        let params = self.parse_go(parts);

        let budget = if params.infinite {
            None
        } else {
            params.movetime.or(params.clock_ms)
        };

        let (release_tx, release_rx) = if params.ponder {
            let (tx, rx) = mpsc::sync_channel(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        self.ponder_tx = release_tx;
        self.pondering = params.ponder;

        let flag = Arc::new(AtomicBool::new(false));
        self.stop_flag = Arc::clone(&flag);

        let board = self.board.clone();
        let depth = params.depth.unwrap_or(64);
        let nodes_limit = params.nodes;
        let time_to_use: Option<u128> = budget;
        let hash_mb = self
            .options
            .get("Hash")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(16);
        let use_book = self
            .options
            .get("Book")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);
        let side_is_white = board.turn == Color::White;

        let handle = thread::spawn(move || {
            let mut searcher = Searcher::new();
            searcher.set_tt_size(hash_mb);
            searcher.set_stop_flag(Arc::clone(&flag));
            if use_book {
                if let Err(e) = searcher.load_opening_book("endgames.epd") {
                    eprintln!("Warning: Failed to load opening book: {}", e);
                }
            }

            let (best_move, achieved_depth) = if let Some(nodes) = nodes_limit {
                let (mv, _) = searcher.search_fixed_nodes(&mut board.clone(), nodes);
                (mv, searcher.stats.search_depth)
            } else {
                let limit = time_to_use.unwrap_or(u128::MAX / 4);
                let watchdog = time_to_use.map(|ms| {
                    let f = Arc::clone(&flag);
                    thread::spawn(move || {
                        thread::sleep(std::time::Duration::from_millis(ms as u64));
                        f.store(true, Ordering::Relaxed);
                    })
                });
                let (mv, achieved) =
                    searcher.search_with_time_management(&mut board.clone(), depth, limit);
                if let Some(wd) = watchdog {
                    let _ = wd.join();
                }
                (mv, achieved)
            };

            if let Some(rx) = release_rx {
                let _ = rx.recv();
            }

            let nodes = searcher.stats.nodes + searcher.stats.qnodes;
            let elapsed_ms = searcher.stats.time_ms.max(1);
            let nps = if elapsed_ms > 0 {
                nodes as u128 * 1000 / elapsed_ms as u128
            } else {
                0
            };

            if let Some(mv) = best_move {
                let score_str = uci_score_string(searcher.best_score, side_is_white);
                println!(
                    "info depth {} score {} nodes {} nps {} time {} pv {}",
                    achieved_depth.max(1),
                    score_str,
                    nodes,
                    nps,
                    elapsed_ms,
                    uci_move_string(&mv)
                );
                println!("bestmove {}", uci_move_string(&mv));
            } else {
                println!("info depth {} nodes {} time {}", achieved_depth.max(1), nodes, elapsed_ms);
                println!("bestmove 0000");
            }
        });

        self.search_thread = Some(handle);
    }

    fn stop_and_join(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(tx) = self.ponder_tx.take() {
            let _ = tx.send(());
        }
        self.pondering = false;
        if let Some(handle) = self.search_thread.take() {
            let _ = handle.join();
        }
    }

    fn cmd_quit(&self) {
        std::process::exit(0);
    }

    fn cmd_perft(&self, parts: &[&str]) {
        if parts.len() < 2 {
            println!("perft requires depth parameter");
            return;
        }

        if let Ok(depth) = parts[1].parse::<u32>() {
            let result = self.board.perft(depth);
            println!("Nodes: {}", result.nodes);
            println!("Captures: {}", result.captures);
            println!("En Passants: {}", result.en_passants);
            println!("Castles: {}", result.castles);
            println!("Promotions: {}", result.promotions);
            println!("Checks: {}", result.checks);
            println!("Checkmates: {}", result.checkmates);
        }
    }
}

fn uci_promo_char(mv: &ChessMove) -> Option<char> {
    match mv.move_type {
        MoveType::Promotion(piece) => Some(piece_to_fen_char(piece).to_ascii_lowercase()),
        _ => None,
    }
}

fn uci_move_string(mv: &ChessMove) -> String {
    let mut s = format!(
        "{}{}",
        Board::square_to_string(mv.from),
        Board::square_to_string(mv.to)
    );
    if let Some(promo) = uci_promo_char(mv) {
        s.push(promo);
    }
    s
}

fn is_uci_move_token(token: &str) -> bool {
    if token.len() != 4 && token.len() != 5 {
        return false;
    }
    let b = token.as_bytes();
    b[0].is_ascii_lowercase() && (b'a'..=b'h').contains(&b[0])
        && (b'1'..=b'8').contains(&b[1])
        && b[2].is_ascii_lowercase()
        && (b'a'..=b'h').contains(&b[2])
        && (b'1'..=b'8').contains(&b[3])
}

fn uci_score_string(score: Score, side_is_white: bool) -> String {
    let from_side_pov = if side_is_white { score } else { -score };
    if from_side_pov.abs() >= 90000 {
        let plies_to_mate = 100000 - from_side_pov.abs();
        let moves_to_mate = (plies_to_mate + 1) / 2;
        format!("mate {}", moves_to_mate)
    } else {
        format!("cp {}", from_side_pov)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uci_creation() {
        let engine = UciEngine::new();
        assert_eq!(engine.info.name, "Rust Chess Engine");
    }

    #[test]
    fn test_position_startpos() {
        let mut engine = UciEngine::new();
        engine.cmd_position(&["position", "startpos"]);

        let moves = engine.board.generate_moves();
        assert_eq!(moves.len(), 20, "Starting position should have 20 legal moves");
    }

    #[test]
    fn test_position_with_moves() {
        let mut engine = UciEngine::new();
        engine.cmd_position(&["position", "startpos", "moves", "e2e4", "e7e5"]);

        let moves = engine.board.generate_moves();
        assert!(!moves.is_empty(), "Position after e2e4 e7e5 should have legal moves");
    }

    #[test]
    fn test_uci_move_notation() {
        let mut engine = UciEngine::new();
        let result = engine.apply_uci_move("e2e4");
        assert!(result.is_ok(), "e2e4 should be a legal move from starting position");
    }

    #[test]
    fn test_uci_castle_notation() {
        let mut engine = UciEngine::new();
        engine.apply_uci_move("e2e4").unwrap();
        engine.apply_uci_move("e7e5").unwrap();
        engine.apply_uci_move("g1f3").unwrap();
        engine.apply_uci_move("b8c6").unwrap();
        engine.apply_uci_move("f1c4").unwrap();
        engine.apply_uci_move("g8f6").unwrap();
        assert!(engine.apply_uci_move("e1g1").is_ok(), "e1g1 castling should be legal");
    }

    #[test]
    fn test_uci_promotion_notation() {
        let mut engine = UciEngine::new();
        engine.cmd_position(&["position", "fen", "8/P7/8/8/8/8/8/k6K w - - 0 1"]);
        assert!(engine.apply_uci_move("a7a8q").is_ok(), "Promotion to queen should be legal");
        engine.cmd_position(&["position", "fen", "8/P7/8/8/8/8/8/k6K w - - 0 1"]);
        assert!(engine.apply_uci_move("a7a8n").is_ok(), "Promotion to knight should be legal");
        engine.cmd_position(&["position", "fen", "8/P7/8/8/8/8/8/k6K w - - 0 1"]);
        assert!(engine.apply_uci_move("a7a8").is_err(), "Bare promotion should be rejected");
    }

    #[test]
    fn test_uci_score_string() {
        assert_eq!(uci_score_string(35, true), "cp 35");
        assert_eq!(uci_score_string(35, false), "cp -35");
        assert_eq!(uci_score_string(-99950, true), "mate 25");
        assert_eq!(uci_score_string(99975, false), "mate 13");
    }

    #[test]
    fn test_uci_move_token_check() {
        assert!(is_uci_move_token("e2e4"));
        assert!(is_uci_move_token("a7a8q"));
        assert!(!is_uci_move_token("searchmoves"));
        assert!(!is_uci_move_token("e2"));
    }
}