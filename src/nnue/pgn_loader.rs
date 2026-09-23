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
    /// Load positions from a PGN file. Phase-balanced endgame-biased sampling:
    /// for each game of at least 45 plies, keeps 1 position, rotating between
    /// opening (ply 10), middlegame (ply 20), late (ply n-12) and terminal
    /// (ply n-1) phases. Scanning stops once `max_positions` is collected.
    pub fn load_pgn<P: AsRef<Path>>(path: P, max_positions: Option<usize>) -> Result<Vec<GamePosition>, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open PGN file: {}", e))?;

        let reader = BufReader::new(file);
        let mut output = Vec::new();
        let mut game_positions: Vec<GamePosition> = Vec::new();
        let mut board = Board::new();
        let mut game_broken = false;
        let mut current_result = 0.5;
        let mut accepted_games = 0usize;

        macro_rules! flush {
            () => {
                if !game_broken
                    && Self::flush_game(
                        &game_positions,
                        &mut output,
                        max_positions,
                        &mut accepted_games,
                    )
                {
                    return Ok(output);
                }
            };
        }

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Error reading file: {}", e))?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if line.starts_with("[") {
                if line.starts_with("[Result") {
                    flush!();
                    game_positions.clear();
                    board = Board::new();
                    game_broken = false;
                    if let Some(result_str) = Self::extract_result(line) {
                        current_result = Self::parse_result(&result_str);
                    }
                }
            } else if !game_broken {
                for tok in line.split_whitespace() {
                    if Self::is_skip_token(tok) {
                        continue;
                    }
                    match Self::apply_move(&mut board, tok) {
                        Ok(()) => game_positions.push(GamePosition {
                            fen: board.to_fen(),
                            result: current_result,
                        }),
                        Err(_) => {
                            game_broken = true;
                            break;
                        }
                    }
                }
            }
        }

        flush!();

        if let Some(max) = max_positions {
            output.truncate(max);
        }
        Ok(output)
    }

    /// Sample 1 position per game, rotating the sampled phase so the dataset
/// covers openings, middlegames and endgames evenly:
///   phase 0: after ply 10 (opening), phase 1: after ply 20 (middlegame),
///   phase 2: ply n-12 (late), phase 3: the final ply n-1 (terminal).
/// `accepted_games` counts games that met the length threshold and rotates
/// the phase choice. Returns `true` when the cap has been reached.
    fn flush_game(
        game_positions: &[GamePosition],
        output: &mut Vec<GamePosition>,
        max_positions: Option<usize>,
        accepted_games: &mut usize,
    ) -> bool {
        let n = game_positions.len();
        if n >= 45 {
            if let Some(max) = max_positions {
                if output.len() >= max {
                    return true;
                }
            }
            let phase = *accepted_games % 4;
            let idx = match phase {
                0 => 9,
                1 => 19,
                2 => n - 12,
                _ => n - 1,
            };
            if let Some(pos) = game_positions.get(idx) {
                *accepted_games += 1;
                output.push(pos.clone());
            }
        }
        if let Some(max) = max_positions {
            output.len() >= max
        } else {
            false
        }
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

    /// Tokens that are move numbers ("1.", "1..."), comments ("{..}"), NAGs
    /// ("$4"), or the trailing result ("1-0", "1/2-1/2"). Castling ("O-O")
    /// is not skipped here; it is handled by `apply_move`.
    fn is_skip_token(tok: &str) -> bool {
        let chars: Vec<char> = tok.chars().collect();
        if chars.is_empty() {
            return true;
        }
        let first = chars[0];
        if tok.contains('.') {
            return true;
        }
        if first == '{' || first == '}' || first == '$' || first == '[' {
            return true;
        }
        if tok == "*" {
            return true;
        }
        if (tok.contains('-') || tok.contains('/')) && chars[0].is_digit(10) {
            return true;
        }
        false
    }

    /// Resolve a SAN move token against the current board. Matches piece
    /// letter, destination square, disambiguation and capture marker so the
    /// board cannot desync from the game transcript.
    fn apply_move(board: &mut Board, move_str: &str) -> Result<(), String> {
        let raw = move_str.trim();

        let norm = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != 'O' && c != '-');
        if norm == "O-O" || norm == "0-0" {
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

        if norm == "O-O-O" || norm == "0-0-0" {
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

        let token = raw.trim_end_matches(|c| matches!(c, '+' | '#' | '!' | '?'));
        let eq_piece = token.find('=').map(|i| token[i + 1..].trim().chars().next());
        let base = match token.find('=') {
            Some(i) => &token[..i],
            None => token,
        };
        if base.is_empty() {
            return Err(format!("Empty move token: {}", move_str));
        }

        let had_capture = base.contains('x');
        let had_capture = had_capture; // lenient: only reject if 'x' forbids a non-capture later
        let clean: String = base.chars().filter(|&c| c != 'x').collect();
        let chars: Vec<char> = clean.chars().collect();
        if chars.len() < 2 {
            return Err(format!("Malformed move token: {}", move_str));
        }

        let (piece, rest) = match chars.first() {
            Some(&c) if "PNBRQK".contains(c) => (Some(c), &chars[1..]),
            _ => (None, chars.as_slice()),
        };

        let dest_file = *rest.get(rest.len() - 2).unwrap();
        let dest_rank = *rest.last().unwrap();
        let dest = format!("{}{}", dest_file, dest_rank);
        let disam: &[char] = &rest[..rest.len() - 2];
        let disam_file: Vec<char> = disam.iter().copied().filter(|c| ('a'..='h').contains(c)).collect();
        let disam_rank: Vec<char> = disam.iter().copied().filter(|c| ('1'..='8').contains(c)).collect();

        let moves = board.generate_moves();
        let mut candidates = Vec::new();

        for mv in moves {
            let from_str = Board::square_to_string(mv.from);
            let to_str = Board::square_to_string(mv.to);
            if to_str != dest {
                continue;
            }

            let is_prom = matches!(mv.move_type, crate::board::chess_move::MoveType::Promotion(_));
            if is_prom != eq_piece.is_some() {
                continue;
            }

            let sq_piece = board.squares[mv.from.0][mv.from.1];
            if piece.is_none() {
                if !matches!(sq_piece, crate::board::pieces::Piece::WhitePawn | crate::board::pieces::Piece::BlackPawn) {
                    continue;
                }
            } else {
                let letter = match sq_piece {
                    crate::board::pieces::Piece::WhiteKnight | crate::board::pieces::Piece::BlackKnight => 'N',
                    crate::board::pieces::Piece::WhiteBishop | crate::board::pieces::Piece::BlackBishop => 'B',
                    crate::board::pieces::Piece::WhiteRook | crate::board::pieces::Piece::BlackRook => 'R',
                    crate::board::pieces::Piece::WhiteQueen | crate::board::pieces::Piece::BlackQueen => 'Q',
                    crate::board::pieces::Piece::WhiteKing | crate::board::pieces::Piece::BlackKing => 'K',
                    _ => '?',
                };
                if letter != piece.unwrap() {
                    continue;
                }
            }

            if !disam_file.is_empty()
                && !disam_file.contains(&from_str.chars().next().unwrap())
            {
                continue;
            }
            if !disam_rank.is_empty()
                && !disam_rank.contains(&from_str.chars().nth(1).unwrap())
            {
                continue;
            }

            let target = board.squares[mv.to.0][mv.to.1];
            let is_capture = target != crate::board::pieces::Piece::Empty
                || matches!(mv.move_type, crate::board::chess_move::MoveType::EnPassant);
            if had_capture && !is_capture {
                continue;
            }

            candidates.push(mv);
        }

        for mv in candidates {
            let from_str = Board::square_to_string(mv.from);
            let to_str = Board::square_to_string(mv.to);
            if let Ok(_) = board.make_move(&from_str, &to_str) {
                return Ok(());
            }
        }

        Err(format!("Move not found: {}", move_str))
    }
}
