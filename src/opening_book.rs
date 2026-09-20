use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::board::{Board, Color};
use crate::board::chess_move::ChessMove;
use crate::board::pieces::Piece;

#[derive(Clone, Debug)]
pub struct BookMove {
    pub move_obj: ChessMove,
    pub weight: u32,
}

#[derive(Clone)]
pub struct OpeningBook {
    positions: HashMap<u64, Vec<BookMove>>,
    loaded: bool,
}

impl OpeningBook {
    pub fn new() -> Self {
        OpeningBook {
            positions: HashMap::new(),
            loaded: false,
        }
    }
    pub fn load_epd<P: AsRef<Path>>(&mut self, path: P) -> Result<usize, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open EPD file: {}", e))?;

        let reader = BufReader::new(file);
        let mut count = 0;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Error reading EPD file: {}", e))?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if self.parse_epd_line(line)? {
                count += 1;
            }
        }

        self.loaded = true;
        Ok(count)
    }

    fn parse_epd_line(&mut self, line: &str) -> Result<bool, String> {
        let parts: Vec<&str> = line.split(';').map(|p| p.trim()).collect();

        if parts.is_empty() {
            return Ok(false);
        }
        let fen_part = parts[0].trim();
        let board = Board::from_fen(fen_part)
            .map_err(|e| format!("Invalid FEN in EPD: {}", e))?;

        let mut moves = Vec::new();

        for part in parts.iter().skip(1) {
            let tokens: Vec<&str> = part.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            let field = tokens[0];
            let values = &tokens[1..];
            if field == "bm" {
                for move_san in values {
                    if let Ok(chess_move) = self.parse_san_move(&board, move_san) {
                        moves.push(BookMove {
                            move_obj: chess_move,
                            weight: 100, 
                        });
                    }
                }
            }

            if field == "ce" && !values.is_empty() {
            }
        }

        if moves.is_empty() {
            return Ok(false);
        }

        let hash = self.position_hash(&board);
        self.positions
            .entry(hash)
            .or_insert_with(Vec::new)
            .extend(moves);

        Ok(true)
    }

    fn parse_san_move(&self, board: &Board, san: &str) -> Result<ChessMove, String> {
        let san = san.trim();

        let san = san.trim_end_matches(&['#', '+'][..]);

        if san == "O-O" || san == "0-0" {
            let king_pos = self.find_king(board);
            return Ok(ChessMove::castling(
                king_pos.0,
                king_pos.1,
                king_pos.0,
                king_pos.1 + 2,
            ));
        }

        if san == "O-O-O" || san == "0-0-0" {
            let king_pos = self.find_king(board);
            return Ok(ChessMove::castling(
                king_pos.0,
                king_pos.1,
                king_pos.0,
                king_pos.1 - 2,
            ));
        }

        let chars: Vec<char> = san.chars().collect();

        let is_pawn = if chars[0].is_lowercase() && chars[0] != 'x' {
            true
        } else if !chars[0].is_uppercase() {
            true
        } else {
            false
        };
        let target_end = if san.contains('=') {
            san.find('=').unwrap()
        } else {
            san.len()
        };

        let target_str = &san[target_end.saturating_sub(2)..target_end];
        if target_str.len() != 2 {
            return Err(format!("Invalid move notation: {}", san));
        }

        let (target_rank, target_file) = self.parse_square_notation(target_str)?;
        let moves = board.generate_moves();

        for chess_move in moves {
            if chess_move.to != (target_rank, target_file) {
                continue;
            }

            if is_pawn && chars.len() > 2 && chars[0].is_lowercase() && chars[0] != 'x' {
                let source_file = self.file_to_index(chars[0]);
                if chess_move.from.1 != source_file {
                    continue;
                }
            }

            return Ok(chess_move);
        }

        Err(format!("Could not parse move: {}", san))
    }

    /// Finds the king position for a given color on the board
    fn find_king(&self, board: &Board) -> (usize, usize) {
        let king_piece = match board.turn {
            Color::White => Piece::WhiteKing,
            Color::Black => Piece::BlackKing,
        };

        for rank in 0..8 {
            for file in 0..8 {
                if board.squares[rank][file] == king_piece {
                    return (rank, file);
                }
            }
        }

        panic!("King not found on board");
    }

    fn parse_square_notation(&self, square: &str) -> Result<(usize, usize), String> {
        let chars: Vec<char> = square.chars().collect();
        if chars.len() != 2 {
            return Err(format!("Invalid square notation: {}", square));
        }

        let file = self.file_to_index(chars[0]);
        let rank = match chars[1] {
            '1'..='8' => {
                let r = (chars[1] as usize) - ('1' as usize);
                7 - r
            }
            _ => return Err(format!("Invalid rank: {}", chars[1])),
        };

        Ok((rank, file))
    }
    fn file_to_index(&self, file: char) -> usize {
        match file {
            'a' => 0,
            'b' => 1,
            'c' => 2,
            'd' => 3,
            'e' => 4,
            'f' => 5,
            'g' => 6,
            'h' => 7,
            _ => 0,
        }
    }

    pub fn position_hash(&self, board: &Board) -> u64 {
        let mut hash: u64 = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                if piece != Piece::Empty {
                    let piece_hash = self.zobrist_piece(piece, rank, file);
                    hash ^= piece_hash;
                }
            }
        }

        if board.turn == Color::Black {
            hash ^= 0x8f1d3642e6842609;
        }
        let castling_hash = self.zobrist_castling(&board.castling_rights);
        hash ^= castling_hash;

        if let Some((ep_rank, ep_file)) = board.en_passant_square {
            let ep_hash = self.zobrist_en_passant(ep_rank, ep_file);
            hash ^= ep_hash;
        }

        hash
    }

    fn zobrist_piece(&self, piece: Piece, rank: usize, file: usize) -> u64 {
        let piece_id = match piece {
            Piece::WhitePawn => 0,
            Piece::WhiteKnight => 1,
            Piece::WhiteBishop => 2,
            Piece::WhiteRook => 3,
            Piece::WhiteQueen => 4,
            Piece::WhiteKing => 5,
            Piece::BlackPawn => 6,
            Piece::BlackKnight => 7,
            Piece::BlackBishop => 8,
            Piece::BlackRook => 9,
            Piece::BlackQueen => 10,
            Piece::BlackKing => 11,
            Piece::Empty => return 0,
        };

        let square_idx = rank * 8 + file;
        let mut hash = 0x9e3779b97f4a7c15u64;
        hash = hash.wrapping_mul((piece_id as u64).wrapping_add(1)).wrapping_mul(0xbf58476d1ce4e5b9);
        hash ^= (square_idx as u64).wrapping_mul(0x94d049bb133111eb);
        hash
    }

    fn zobrist_castling(&self, rights: &crate::board::CastlingRights) -> u64 {
        let mut hash = 0u64;
        if rights.white_kingside {
            hash ^= 0x1234567890abcdef;
        }
        if rights.white_queenside {
            hash ^= 0x2345678901bcdef0;
        }
        if rights.black_kingside {
            hash ^= 0x3456789012cdef01;
        }
        if rights.black_queenside {
            hash ^= 0x456789023def012;
        }
        hash
    }

    fn zobrist_en_passant(&self, rank: usize, file: usize) -> u64 {
        0xabcdef0123456789u64.wrapping_mul(((rank * 8 + file) as u64).wrapping_add(1))
    }

    pub fn get_book_move(&self, board: &Board) -> Option<ChessMove> {
        let hash = self.position_hash(board);

        self.positions.get(&hash).and_then(|moves| {
            moves.iter().max_by_key(|m| m.weight).map(|m| m.move_obj)
        })
    }

    pub fn get_all_book_moves(&self, board: &Board) -> Option<Vec<BookMove>> {
        let hash = self.position_hash(board);
        self.positions.get(&hash).cloned()
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn size(&self) -> usize {
        self.positions.len()
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.loaded = false;
    }

    pub fn contains_position(&self, board: &Board) -> bool {
        let hash = self.position_hash(board);
        self.positions.contains_key(&hash)
    }
}

impl Default for OpeningBook {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opening_book_creation() {
        let book = OpeningBook::new();
        assert!(!book.is_loaded());
        assert_eq!(book.size(), 0);
    }

    #[test]
    fn test_opening_book_basic() {
        let book = OpeningBook::new();
        let board = Board::new();

        // Initially not in book
        assert!(!book.contains_position(&board));
        assert!(book.get_book_move(&board).is_none());
    }

    #[test]
    fn test_position_hash_consistency() {
        let book = OpeningBook::new();
        let board = Board::new();

        // Same position should give same hash
        let hash1 = book.position_hash(&board);
        let hash2 = book.position_hash(&board);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_parse_square_notation() {
        let book = OpeningBook::new();

        let (rank, file) = book.parse_square_notation("e4").unwrap();
        assert_eq!(file, 4); // e-file
        assert_eq!(rank, 4); // 4th rank (0-indexed from top)

        let (rank, file) = book.parse_square_notation("a1").unwrap();
        assert_eq!(file, 0); // a-file
        assert_eq!(rank, 7); // 1st rank (0-indexed from top)

        assert!(book.parse_square_notation("z9").is_err());
        assert!(book.parse_square_notation("e").is_err());
    }

    #[test]
    fn test_file_to_index() {
        let book = OpeningBook::new();

        assert_eq!(book.file_to_index('a'), 0);
        assert_eq!(book.file_to_index('e'), 4);
        assert_eq!(book.file_to_index('h'), 7);
    }

    #[test]
    fn test_castling_hash() {
        let book = OpeningBook::new();
        let mut board1 = Board::new();

        let hash1 = book.position_hash(&board1);

        board1.castling_rights.white_kingside = false;
        let hash2 = book.position_hash(&board1);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_en_passant_hash() {
        let book = OpeningBook::new();
        let mut board1 = Board::new();

        let hash1 = book.position_hash(&board1);

        board1.en_passant_square = Some((4, 4));
        let hash2 = book.position_hash(&board1);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_turn_hash() {
        let book = OpeningBook::new();
        let board1 = Board::new();
        let mut board2 = Board::new();

        board2.turn = Color::Black;

        let hash1 = book.position_hash(&board1);
        let hash2 = book.position_hash(&board2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_book_move_struct() {
        let move_obj = ChessMove::new(6, 4, 4, 4);
        let book_move = BookMove {
            move_obj,
            weight: 50,
        };

        assert_eq!(book_move.weight, 50);
        assert_eq!(book_move.move_obj.from, (6, 4));
        assert_eq!(book_move.move_obj.to, (4, 4));
    }

    #[test]
    fn test_multiple_moves_same_position() {
        let mut book = OpeningBook::new();
        let hash = 0x123456789abcdef0u64;

        let move1 = BookMove {
            move_obj: ChessMove::new(6, 4, 4, 4),
            weight: 100,
        };
        let move2 = BookMove {
            move_obj: ChessMove::new(6, 3, 4, 3),
            weight: 80,
        };

        book.positions.insert(hash, vec![move1, move2]);

        let moves = book.positions.get(&hash);
        assert!(moves.is_some());
        assert_eq!(moves.unwrap().len(), 2);
    }

    #[test]
    fn test_book_clear() {
        let mut book = OpeningBook::new();
        book.positions.insert(0x123456789abcdef0u64, vec![]);
        assert_eq!(book.size(), 1);

        book.clear();
        assert_eq!(book.size(), 0);
        assert!(!book.is_loaded());
    }
}
