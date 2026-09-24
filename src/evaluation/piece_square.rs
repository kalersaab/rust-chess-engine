use crate::board::Board;
use crate::board::pieces::Piece;
use crate::evaluation::Score;

pub struct PieceSquareEvaluation;

impl PieceSquareEvaluation {
    pub fn evaluate(board: &Board) -> Score {
        let phase = Self::game_phase(board);

        let mut white_score = 0;
        let mut black_score = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                let value = Self::piece_square_value(piece, rank, file, phase);
                
                if crate::board::pieces::is_white(piece) {
                    white_score += value;
                } else if crate::board::pieces::is_black(piece) {
                    black_score += value;
                }
            }
        }

        let from_white = if board.turn == crate::board::Color::White {
            white_score - black_score
        } else {
            black_score - white_score
        };

        from_white
    }

    /// 1.0 in the opening, tapering to 0.0 as non-pawn material disappears.
    fn game_phase(board: &Board) -> f64 {
        let mut non_pawn = 0u32;
        for rank in 0..8 {
            for file in 0..8 {
                match board.squares[rank][file] {
                    Piece::WhiteKnight | Piece::BlackKnight => non_pawn += 320,
                    Piece::WhiteBishop | Piece::BlackBishop => non_pawn += 330,
                    Piece::WhiteRook | Piece::BlackRook => non_pawn += 500,
                    Piece::WhiteQueen | Piece::BlackQueen => non_pawn += 900,
                    _ => {}
                }
            }
        }
        let full = 6400.0;
        (non_pawn as f64 / full).clamp(0.0, 1.0)
    }

    fn piece_square_value(piece: Piece, rank: usize, file: usize, phase: f64) -> Score {
        match piece {
            Piece::WhitePawn => Self::white_pawn_table()[rank][file],
            Piece::WhiteKnight => Self::white_knight_table()[rank][file],
            Piece::WhiteBishop => Self::white_bishop_table()[rank][file],
            Piece::WhiteRook => Self::white_rook_table()[rank][file],
            Piece::WhiteQueen => Self::white_queen_table()[rank][file],
            Piece::WhiteKing => {
                Self::blend(
                    Self::white_king_table()[rank][file],
                    Self::white_king_endgame_table()[rank][file],
                    phase,
                )
            }
            
            Piece::BlackPawn => Self::black_pawn_table()[rank][file],
            Piece::BlackKnight => Self::black_knight_table()[rank][file],
            Piece::BlackBishop => Self::black_bishop_table()[rank][file],
            Piece::BlackRook => Self::black_rook_table()[rank][file],
            Piece::BlackQueen => Self::black_queen_table()[rank][file],
            Piece::BlackKing => {
                Self::blend(
                    Self::black_king_table()[rank][file],
                    Self::black_king_endgame_table()[rank][file],
                    phase,
                )
            }
            
            Piece::Empty => 0,
        }
    }

    /// midgame->endgame taper: value = mid*phase + end*(1-phase).
    fn blend(mid: Score, end: Score, phase: f64) -> Score {
        (mid as f64 * phase + end as f64 * (1.0 - phase)).round() as Score
    }

    fn white_pawn_table() -> [[Score; 8]; 8] {
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5, 5, 10, 25, 25, 10, 5, 5],
            [0, 0, 0, 20, 20, 0, 0, 0],
            [5, -5, -10, 0, 0, -10, -5, 5],
            [5, 10, 10, -20, -20, 10, 10, 5],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn black_pawn_table() -> [[Score; 8]; 8] {
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 10, 10, -20, -20, 10, 10, 5],
            [5, -5, -10, 0, 0, -10, -5, 5],
            [0, 0, 0, 20, 20, 0, 0, 0],
            [5, 5, 10, 25, 25, 10, 5, 5],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn white_knight_table() -> [[Score; 8]; 8] {
        [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20, 0, 0, 0, 0, -20, -40],
            [-30, 0, 10, 15, 15, 10, 0, -30],
            [-30, 5, 15, 20, 20, 15, 5, -30],
            [-30, 0, 15, 20, 20, 15, 0, -30],
            [-30, 5, 10, 15, 15, 10, 5, -30],
            [-40, -20, 0, 5, 5, 0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50],
        ]
    }

    fn black_knight_table() -> [[Score; 8]; 8] {
        [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20, 0, 5, 5, 0, -20, -40],
            [-30, 5, 10, 15, 15, 10, 5, -30],
            [-30, 0, 15, 20, 20, 15, 0, -30],
            [-30, 5, 15, 20, 20, 15, 5, -30],
            [-30, 0, 10, 15, 15, 10, 0, -30],
            [-40, -20, 0, 0, 0, 0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50],
        ]
    }

    fn white_bishop_table() -> [[Score; 8]; 8] {
        [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 10, 10, 5, 0, -10],
            [-10, 5, 5, 10, 10, 5, 5, -10],
            [-10, 0, 10, 10, 10, 10, 0, -10],
            [-10, 10, 10, 10, 10, 10, 10, -10],
            [-10, 5, 0, 0, 0, 0, 5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20],
        ]
    }

    fn black_bishop_table() -> [[Score; 8]; 8] {
        [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10, 5, 0, 0, 0, 0, 5, -10],
            [-10, 10, 10, 10, 10, 10, 10, -10],
            [-10, 0, 10, 10, 10, 10, 0, -10],
            [-10, 5, 5, 10, 10, 5, 5, -10],
            [-10, 0, 5, 10, 10, 5, 0, -10],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20],
        ]
    }

    fn white_rook_table() -> [[Score; 8]; 8] {
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 10, 10, 10, 10, 10, 10, 5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [0, 0, 0, 5, 5, 0, 0, 0],
        ]
    }

    fn black_rook_table() -> [[Score; 8]; 8] {
        [
            [0, 0, 0, 5, 5, 0, 0, 0],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [-5, 0, 0, 0, 0, 0, 0, -5],
            [5, 10, 10, 10, 10, 10, 10, 5],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn white_queen_table() -> [[Score; 8]; 8] {
        [
            [-20, -10, -10, -5, -5, -10, -10, -20],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-10, 0, 5, 5, 5, 5, 0, -10],
            [-5, 0, 5, 5, 5, 5, 0, -5],
            [0, 0, 5, 5, 5, 5, 0, -5],
            [-10, 5, 5, 5, 5, 5, 0, -10],
            [-10, 0, 5, 0, 0, 0, 0, -10],
            [-20, -10, -10, -5, -5, -10, -10, -20],
        ]
    }

    fn black_queen_table() -> [[Score; 8]; 8] {
        [
            [-20, -10, -10, -5, -5, -10, -10, -20],
            [-10, 0, 5, 0, 0, 0, 0, -10],
            [-10, 5, 5, 5, 5, 5, 0, -10],
            [0, 0, 5, 5, 5, 5, 0, -5],
            [-5, 0, 5, 5, 5, 5, 0, -5],
            [-10, 0, 5, 5, 5, 5, 0, -10],
            [-10, 0, 0, 0, 0, 0, 0, -10],
            [-20, -10, -10, -5, -5, -10, -10, -20],
        ]
    }

    fn white_king_table() -> [[Score; 8]; 8] {
        [
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [20, 30, 10, 0, 0, 10, 30, 20],
            [20, 30, 30, 10, 10, 30, 30, 20],
        ]
    }

    fn black_king_table() -> [[Score; 8]; 8] {
        [
            [20, 30, 30, 10, 10, 30, 30, 20],
            [20, 30, 10, 0, 0, 10, 30, 20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
        ]
    }

    /// Endgame: the king should march toward the centre rather than stay
    /// parked on its castling square. Blended in as material disappears.
    fn white_king_endgame_table() -> [[Score; 8]; 8] {
        [
            [-24, -48, -72, -72, -72, -72, -48, -24],
            [-48, -64, -16, -24, -24, -16, -64, -48],
            [-72, -16, 0, 8, 8, 0, -16, -72],
            [-72, -24, 8, 40, 40, 8, -24, -72],
            [-72, -24, 8, 40, 40, 8, -24, -72],
            [-72, -16, 0, 8, 8, 0, -16, -72],
            [-48, -64, -16, -24, -24, -16, -64, -48],
            [-24, -48, -72, -72, -72, -72, -48, -24],
        ]
    }

    fn black_king_endgame_table() -> [[Score; 8]; 8] {
        [
            [-24, -48, -72, -72, -72, -72, -48, -24],
            [-48, -64, -16, -24, -24, -16, -64, -48],
            [-72, -16, 0, 8, 8, 0, -16, -72],
            [-72, -24, 8, 40, 40, 8, -24, -72],
            [-72, -24, 8, 40, 40, 8, -24, -72],
            [-72, -16, 0, 8, 8, 0, -16, -72],
            [-48, -64, -16, -24, -24, -16, -64, -48],
            [-24, -48, -72, -72, -72, -72, -48, -24],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_king_endgame_table_centralizes() {
        assert!(
            PieceSquareEvaluation::white_king_endgame_table()[4][4]
                > PieceSquareEvaluation::white_king_endgame_table()[7][7],
            "endgame king table should reward centralization over the corner"
        );
    }

    #[test]
    fn test_phase_blend_taper() {
        assert_eq!(PieceSquareEvaluation::blend(20, 40, 0.0), 40);
        assert_eq!(PieceSquareEvaluation::blend(20, 40, 1.0), 20);
        assert_eq!(PieceSquareEvaluation::blend(20, 40, 0.5), 30);
    }

    #[test]
    fn test_game_phase_detects_endgame() {
        // Kings only: no non-pawn material left.
        let board = Board::from_fen("8/8/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        assert!(PieceSquareEvaluation::game_phase(&board) < 0.01);
    }

    #[test]
    fn test_pawn_advanced_scores_higher() {
        // A pawn far up the board (rank 2, index 1) beats one still at rank 5 territory.
        assert!(
            PieceSquareEvaluation::white_pawn_table()[1][1]
                > PieceSquareEvaluation::white_pawn_table()[4][1]
        );
    }
}
