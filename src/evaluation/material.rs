use crate::board::Board;
use crate::board::pieces::Piece;
use crate::evaluation::Score;

pub struct MaterialEvaluation;

impl MaterialEvaluation {
    pub fn evaluate(board: &Board) -> Score {
        let mut white_material = 0;
        let mut black_material = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                match piece {
                    Piece::WhitePawn => white_material += 100,
                    Piece::WhiteKnight => white_material += 320,
                    Piece::WhiteBishop => white_material += 330,
                    Piece::WhiteRook => white_material += 500,
                    Piece::WhiteQueen => white_material += 900,
                    Piece::WhiteKing => white_material += 0,
                    
                    Piece::BlackPawn => black_material += 100,
                    Piece::BlackKnight => black_material += 320,
                    Piece::BlackBishop => black_material += 330,
                    Piece::BlackRook => black_material += 500,
                    Piece::BlackQueen => black_material += 900,
                    Piece::BlackKing => black_material += 0,
                    
                    Piece::Empty => {}
                }
            }
        }

        let from_white = if board.turn == crate::board::Color::White {
            white_material as Score - black_material as Score
        } else {
            black_material as Score - white_material as Score
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
