use crate::board::{Board, Color};
use crate::board::pieces::Piece;
use crate::evaluation::Score;

pub struct PieceMobility;

impl PieceMobility {
    pub fn evaluate(board: &Board) -> Score {
        let mut white_mobility = 0;
        let mut black_mobility = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                let moves_count = Self::count_moves(board, (rank, file), piece);

                match piece {
                    Piece::WhiteKnight | Piece::WhiteBishop | Piece::WhiteRook | Piece::WhiteQueen => {
                        white_mobility += moves_count as Score;
                    }
                    Piece::BlackKnight | Piece::BlackBishop | Piece::BlackRook | Piece::BlackQueen => {
                        black_mobility += moves_count as Score;
                    }
                    _ => {}
                }
            }
        }

        (white_mobility - black_mobility) * 2
    }

    fn count_moves(board: &Board, square: (usize, usize), piece: Piece) -> usize {
        if piece == Piece::Empty {
            return 0;
        }

        let mut count = 0;
        let moves = board.generate_moves();

        for mv in moves {
            if mv.from == square {
                count += 1;
            }
        }

        count
    }
}

pub struct KingSafety;

impl KingSafety {
    pub fn evaluate(board: &Board) -> Score {
        let white_safety = Self::evaluate_king_safety(board, Color::White);
        let black_safety = Self::evaluate_king_safety(board, Color::Black);

        white_safety - black_safety
    }

    fn evaluate_king_safety(board: &Board, color: Color) -> Score {
        let king_piece = match color {
            Color::White => Piece::WhiteKing,
            Color::Black => Piece::BlackKing,
        };

        let mut king_pos = (0, 0);
        for rank in 0..8 {
            for file in 0..8 {
                if board.squares[rank][file] == king_piece {
                    king_pos = (rank, file);
                    break;
                }
            }
        }

        let pawn_shield = Self::count_pawn_shield(board, king_pos, color);
        let exposed = Self::is_king_exposed(board, king_pos, color);

        let mut score = pawn_shield * 10;
        if exposed {
            score -= 50;
        }

        score
    }

    fn count_pawn_shield(board: &Board, king_pos: (usize, usize), color: Color) -> Score {
        let pawn_piece = match color {
            Color::White => Piece::WhitePawn,
            Color::Black => Piece::BlackPawn,
        };

        let mut count = 0;

        for rank_offset in 0..=2 {
            for file_offset in -1..=1 {
                let rank = king_pos.0 as i32 + rank_offset as i32;
                let file = king_pos.1 as i32 + file_offset;

                if rank >= 0 && rank < 8 && file >= 0 && file < 8 {
                    if board.squares[rank as usize][file as usize] == pawn_piece {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn is_king_exposed(board: &Board, king_pos: (usize, usize), color: Color) -> bool {
        let opponent_color = match color {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        let moves = board.generate_moves();
        for mv in moves {
            let piece = board.squares[mv.from.0][mv.from.1];
            if Self::piece_color(piece) == opponent_color {
                if Self::attacks_square(board, mv.from, king_pos) {
                    return true;
                }
            }
        }

        false
    }

    fn attacks_square(_board: &Board, from: (usize, usize), to: (usize, usize)) -> bool {
        let rank_diff = (from.0 as i32 - to.0 as i32).abs();
        let file_diff = (from.1 as i32 - to.1 as i32).abs();

        if rank_diff <= 2 && file_diff <= 2 {
            return true;
        }

        false
    }

    fn piece_color(piece: Piece) -> Color {
        match piece {
            Piece::WhitePawn | Piece::WhiteKnight | Piece::WhiteBishop | 
            Piece::WhiteRook | Piece::WhiteQueen | Piece::WhiteKing => Color::White,
            Piece::BlackPawn | Piece::BlackKnight | Piece::BlackBishop | 
            Piece::BlackRook | Piece::BlackQueen | Piece::BlackKing => Color::Black,
            Piece::Empty => Color::White,
        }
    }
}

pub struct PawnStructure;

impl PawnStructure {
    pub fn evaluate(board: &Board) -> Score {
        let white_score = Self::evaluate_pawns(board, Piece::WhitePawn);
        let black_score = Self::evaluate_pawns(board, Piece::BlackPawn);

        white_score - black_score
    }

    fn evaluate_pawns(board: &Board, pawn_piece: Piece) -> Score {
        let mut score = 0;

        for rank in 0..8 {
            for file in 0..8 {
                if board.squares[rank][file] == pawn_piece {
                    if Self::is_isolated_pawn(board, (rank, file), pawn_piece) {
                        score -= 20;
                    }

                    if Self::is_doubled_pawn(board, (rank, file), pawn_piece) {
                        score -= 15;
                    }

                    if Self::is_passed_pawn(board, (rank, file), pawn_piece) {
                        score += 30;
                    }

                    if Self::is_backward_pawn(board, (rank, file), pawn_piece) {
                        score -= 10;
                    }
                }
            }
        }

        score
    }

    fn is_isolated_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        let adjacent_files = [
            if pos.1 > 0 { Some(pos.1 - 1) } else { None },
            if pos.1 < 7 { Some(pos.1 + 1) } else { None },
        ];

        for file_opt in adjacent_files.iter() {
            if let Some(file) = file_opt {
                for rank in 0..8 {
                    if board.squares[rank][*file] == pawn_piece {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn is_doubled_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        for rank in 0..8 {
            if rank != pos.0 && board.squares[rank][pos.1] == pawn_piece {
                return true;
            }
        }
        false
    }

    fn is_passed_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        let is_white = matches!(pawn_piece, Piece::WhitePawn);
        let opponent_pawn = if is_white { Piece::BlackPawn } else { Piece::WhitePawn };

        let start_rank = if is_white { pos.0 + 1 } else { if pos.0 > 0 { pos.0 - 1 } else { 0 } };
        let end_rank = if is_white { 8 } else { 0 };

        for file_offset in -1..=1 {
            let file = pos.1 as i32 + file_offset;
            if file >= 0 && file < 8 {
                let file = file as usize;
                
                if is_white {
                    for rank in start_rank..end_rank {
                        if board.squares[rank][file] == opponent_pawn {
                            return false;
                        }
                    }
                } else {
                    for rank in (end_rank..start_rank).rev() {
                        if board.squares[rank][file] == opponent_pawn {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    fn is_backward_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        let is_white = matches!(pawn_piece, Piece::WhitePawn);
        let support_rank = if is_white { pos.0 + 1 } else { if pos.0 > 0 { pos.0 - 1 } else { return false; } };

        let support_files = [
            if pos.1 > 0 { Some(pos.1 - 1) } else { None },
            if pos.1 < 7 { Some(pos.1 + 1) } else { None },
        ];

        for file_opt in support_files.iter() {
            if let Some(file) = file_opt {
                if board.squares[support_rank][*file] == pawn_piece {
                    return false;
                }
            }
        }

        true
    }
}
