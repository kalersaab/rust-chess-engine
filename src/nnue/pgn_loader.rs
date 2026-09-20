use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use crate::board::Board;

#[derive(Clone, Debug)]
pub struct GamePosition {
    pub fen: String,
    pub result: f32,
}

pub struct PGNLoader;

impl PGNLoader {
    pub fn load_pgn<P: AsRef<Path>>(path: P, max_positions: Option<usize>) -> Result<Vec<GamePosition>, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open PGN file: {}", e))?;

        let reader = BufReader::new(file);
        let mut positions = Vec::new();
        let mut current_result = 0.5;
        let mut moves = String::new();
        let mut in_game = false;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Error reading file: {}", e))?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if line.starts_with("[Result") {
                if let Some(result_str) = Self::extract_result(line) {
                    current_result = Self::parse_result(&result_str);
                }
            } else if line.starts_with("[") {
                continue;
            } else {
                if !in_game {
                    in_game = true;
                    moves.clear();
                }

                moves.push_str(line);
                moves.push(' ');

                if moves.len() > 2 {
                    if let Ok(game_positions) = Self::extract_positions(&moves, current_result) {
                        positions.extend(game_positions);

                        if let Some(max) = max_positions {
                            if positions.len() >= max {
                                return Ok(positions.into_iter().take(max).collect());
                            }
                        }
                    }
                    in_game = false;
                }
            }
        }

        Ok(positions)
    }

    fn extract_result(line: &str) -> Option<String> {
        if let Some(start) = line.find('\"') {
            if let Some(end) = line[start + 1..].find('\"') {
                return Some(line[start + 1..start + 1 + end].to_string());
            }
        }
        None
    }

    fn parse_result(result: &str) -> f32 {
        match result {
            "1-0" => 1.0,
            "0-1" => 0.0,
            "1/2-1/2" => 0.5,
            _ => 0.5,
        }
    }

    fn extract_positions(moves_str: &str, result: f32) -> Result<Vec<GamePosition>, String> {
        let mut positions = Vec::new();
        let mut board = Board::new();

        let moves: Vec<&str> = moves_str.split_whitespace().collect();

        for move_str in moves {
            if move_str.contains('.') || move_str.contains('-') || move_str.is_empty() {
                continue;
            }

            if Self::is_move_notation(move_str) {
                let fen = board.to_fen();
                positions.push(GamePosition {
                    fen: fen.clone(),
                    result,
                });

                if let Ok(_) = Self::apply_move(&mut board, move_str) {
                    continue;
                } else {
                    break;
                }
            }
        }

        Ok(positions)
    }

    fn is_move_notation(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }

        let chars: Vec<char> = s.chars().collect();

        if s == "O-O" || s == "O-O-O" || s == "0-0" || s == "0-0-0" {
            return true;
        }

        let lowercase_count = chars.iter().filter(|c| c.is_lowercase()).count();
        let uppercase_count = chars.iter().filter(|c| c.is_uppercase()).count();

        (lowercase_count >= 1 && chars.len() >= 2) || 
        (uppercase_count >= 1 && lowercase_count >= 1)
    }

    fn apply_move(board: &mut Board, move_str: &str) -> Result<(), String> {
        let move_str = move_str.trim_matches(|c: char| !c.is_alphanumeric() && c != 'O' && c != '-' && c != 'x');

        if move_str == "O-O" || move_str == "0-0" {
            let king_moves = if board.turn == crate::board::Color::White {
                vec![("e1", "g1"), ("e1", "h1")]
            } else {
                vec![("e8", "g8"), ("e8", "h8")]
            };

            for (from, to) in king_moves {
                if board.make_move(from, to).is_ok() {
                    return Ok(());
                }
            }
            return Err("Invalid castling".to_string());
        }

        if move_str == "O-O-O" || move_str == "0-0-0" {
            let king_moves = if board.turn == crate::board::Color::White {
                vec![("e1", "c1"), ("e1", "a1")]
            } else {
                vec![("e8", "c8"), ("e8", "a8")]
            };

            for (from, to) in king_moves {
                if board.make_move(from, to).is_ok() {
                    return Ok(());
                }
            }
            return Err("Invalid castling".to_string());
        }

        let moves = board.generate_moves();

        for mv in moves {
            let from_str = Board::square_to_string(mv.from);
            let to_str = Board::square_to_string(mv.to);

            if move_str.contains(&from_str) && move_str.contains(&to_str) {
                board.make_move(&from_str, &to_str)?;
                return Ok(());
            }

            if move_str.ends_with(&to_str) {
                if let Ok(_) = board.make_move(&from_str, &to_str) {
                    return Ok(());
                }
            }
        }

        Err(format!("Move not found: {}", move_str))
    }
}
