use super::pieces::Piece;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveType {
    Normal,
    Castling,
    EnPassant,
    Promotion(Piece), // The piece to promote to (Queen, Rook, Bishop, Knight)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChessMove {
    pub from: (usize, usize),
    pub to: (usize, usize),
    pub move_type: MoveType,
}

impl ChessMove {
    pub fn new(
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
    ) -> Self {
        Self {
            from: (from_rank, from_file),
            to: (to_rank, to_file),
            move_type: MoveType::Normal,
        }
    }

    pub fn promotion(
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
        promotion_piece: Piece,
    ) -> Self {
        Self {
            from: (from_rank, from_file),
            to: (to_rank, to_file),
            move_type: MoveType::Promotion(promotion_piece),
        }
    }

    pub fn castling(
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
    ) -> Self {
        Self {
            from: (from_rank, from_file),
            to: (to_rank, to_file),
            move_type: MoveType::Castling,
        }
    }

    pub fn en_passant(
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
    ) -> Self {
        Self {
            from: (from_rank, from_file),
            to: (to_rank, to_file),
            move_type: MoveType::EnPassant,
        }
    }
}