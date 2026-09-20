
pub mod material;
pub mod piece_square;

pub type Score = i32;

use crate::board::Board;

pub struct Evaluator;

impl Evaluator {
    pub fn evaluate(_board: &Board) -> Score {
        0
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
        assert!(score.abs() < 1000);
    }
}
