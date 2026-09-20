
pub mod material;
pub mod piece_square;
pub mod advanced;

pub type Score = i32;

use crate::board::Board;
use material::MaterialEvaluation;
use piece_square::PieceSquareEvaluation;
use advanced::{PieceMobility, KingSafety, PawnStructure};

pub struct Evaluator;

impl Evaluator {
    pub fn evaluate(board: &Board) -> Score {
        let material = MaterialEvaluation::evaluate(board);
        let piece_square = PieceSquareEvaluation::evaluate(board);
        let mobility = PieceMobility::evaluate(board);
        let king_safety = KingSafety::evaluate(board);
        let pawn_structure = PawnStructure::evaluate(board);

        material + piece_square + mobility + king_safety + pawn_structure
    }

    pub fn terminal_score(board: &Board) -> Score {
        if board.is_in_check(board.turn) {
            -100000
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_starting_position_evaluation() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let score = Evaluator::evaluate(&board);
        assert!(score.abs() < 10000);
    }
}
