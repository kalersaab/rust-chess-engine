use crate::board::Board;
use std::collections::HashMap;
use std::io::{self, BufRead};

pub struct EngineInfo {
    pub name: String,
    pub author: String,
}

pub struct UciEngine {
    pub info: EngineInfo,
    pub board: Board,
    pub options: HashMap<String, String>,
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
            "position" => self.cmd_position(&parts),
            "go" => self.cmd_go(&parts),
            "quit" => self.cmd_quit(),
            "display" => self.board.print(),
            "perft" => self.cmd_perft(&parts),
            _ => {
            }
        }
    }

    fn cmd_uci(&self) {
        println!("id name {}", self.info.name);
        println!("id author {}", self.info.author);
        println!("option name Hash type spin default 16 min 1 max 256");
        println!("option name Depth type spin default 6 min 1 max 20");
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
                        }
                    }
                }
                "Depth" => {
                    if let Ok(depth) = value.parse::<u32>() {
                        if depth >= 1 && depth <= 20 {
                            self.options.insert(name.to_string(), value.clone());
                        }
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
                    let mut fen_end = 2;
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
        if move_notation.len() < 4 {
            return Err("Invalid move notation".to_string());
        }

        let from = &move_notation[0..2];
        let to = &move_notation[2..4];

        let legal_moves = self.board.generate_moves();
        let mut found_move = false;

        for legal_move in &legal_moves {
            let move_str = format!(
                "{}{}",
                Board::square_to_string(legal_move.from),
                Board::square_to_string(legal_move.to)
            );

            if move_str == move_notation || move_str == &move_notation[0..4] {
                found_move = true;
                break;
            }
        }

        if !found_move {
            return Err(format!("Illegal move: {}", move_notation));
        }

        if move_notation.len() == 5 {
            let _promotion_char = move_notation.chars().nth(4).unwrap();
            self.board.make_move(from, to)?;
        } else {
            self.board.make_move(from, to)?;
        }

        Ok(())
    }

    pub fn current_fen(&self) -> String {
        self.board.to_fen()
    }

    pub fn get_legal_moves(&self) -> Vec<String> {
        self.board
            .generate_moves()
            .iter()
            .map(|m| {
                format!(
                    "{}{}",
                    Board::square_to_string(m.from),
                    Board::square_to_string(m.to)
                )
            })
            .collect()
    }
    fn cmd_go(&mut self, parts: &[&str]) {
        let mut depth = 6;
        let mut _movetime = None;

        let mut i = 1;
        while i < parts.len() {
            match parts[i] {
                "depth" if i + 1 < parts.len() => {
                    if let Ok(d) = parts[i + 1].parse::<u32>() {
                        depth = d;
                    }
                    i += 2;
                }
                "movetime" if i + 1 < parts.len() => {
                    if let Ok(t) = parts[i + 1].parse::<u32>() {
                        _movetime = Some(t);
                    }
                    i += 2;
                }
                "wtime" | "btime" | "winc" | "binc" => {
                    i += 2;
                }
                "searchmoves" => {
                    i += 1;
                    while i < parts.len() && !parts[i].starts_with(|c: char| !c.is_alphanumeric()) {
                        i += 1;
                    }
                }
                "infinite" => {
                    depth = 20; // Max depth for infinite
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }

        let moves = self.board.generate_moves();

        if !moves.is_empty() {
            let best_move = moves[0];
            let move_notation = format!(
                "{}{}",
                Board::square_to_string(best_move.from),
                Board::square_to_string(best_move.to)
            );
            
            println!("info depth {} seldepth {} nodes 1", depth, depth);
            println!("bestmove {}", move_notation);
        } else {
            println!("bestmove 0000"); 
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
}
