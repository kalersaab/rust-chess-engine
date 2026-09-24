use crate::board::Board;
use crate::board::pieces::Piece;
use crate::evaluation::Score;

pub struct MaterialEvaluation;

impl MaterialEvaluation {
    pub fn evaluate(board: &Board) -> Score {
        let mut white_material = 0;
        let mut black_material = 0;
        let mut white_bishops = 0;
        let mut black_bishops = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                match piece {
                    Piece::WhitePawn => white_material += 100,
                    Piece::WhiteKnight => white_material += 320,
                    Piece::WhiteBishop => {
                        white_material += 330;
                        white_bishops += 1;
                    }
                    Piece::WhiteRook => white_material += 500,
                    Piece::WhiteQueen => white_material += 900,
                    Piece::WhiteKing => white_material += 0,
                    
                    Piece::BlackPawn => black_material += 100,
                    Piece::BlackKnight => black_material += 320,
                    Piece::BlackBishop => {
                        black_material += 330;
                        black_bishops += 1;
                    }
                    Piece::BlackRook => black_material += 500,
                    Piece::BlackQueen => black_material += 900,
                    Piece::BlackKing => black_material += 0,
                    
                    Piece::Empty => {}
                }
            }
        }

        let white_score = white_material + if white_bishops >= 2 { 40 } else { 0 };
        let black_score = black_material + if black_bishops >= 2 { 40 } else { 0 };

        let from_white = if board.turn == crate::board::Color::White {
            white_score as Score - black_score as Score
        } else {
            black_score as Score - white_score as Score
        };

        from_white
    }

    pub fn piece_value(piece: Piece) -> u32 {
        match piece {
            Piece::Empty => 0,
            Piece::WhitePawn | Piece::BlackPawn => 100,
            Piece::WhiteKnight | Piece::BlackKnight => 320,
            Piece::WhiteBishop | Piece::BlackBishop => 330,
            Piece::WhiteRook | Piece::BlackRook => 500,
            Piece::WhiteQueen | Piece::BlackQueen => 900,
            Piece::WhiteKing | Piece::BlackKing => 0,
        }
    }

    pub fn material_count(board: &Board) -> (u32, u32) {
        let mut white = 0;
        let mut black = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                if crate::board::pieces::is_white(piece) {
                    white += Self::piece_value(piece);
                } else if crate::board::pieces::is_black(piece) {
                    black += Self::piece_value(piece);
                }
            }
        }

        (white, black)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bishop_pair_bonus() {
        let board = Board::from_fen("2B1k3/2B5/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(MaterialEvaluation::evaluate(&board), 700);
    }

    #[test]
    fn test_bishop_pair_requires_two_bishops() {
        // A single bishop gets no pair bonus.
        let one = Board::from_fen("2B1k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(MaterialEvaluation::evaluate(&one), 330);
    }
}
