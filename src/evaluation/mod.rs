
pub mod material;
pub mod piece_square;
pub mod advanced;

pub type Score = i32;

use crate::board::Board;
use material::MaterialEvaluation;
use piece_square::PieceSquareEvaluation;
use advanced::{PieceMobility, KingSafety, PawnStructure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationMode {
    Handcrafted,
    NNUE,
    Hybrid,
}

pub struct Evaluator {
    pub nnue_network: Option<crate::nnue::NNUENetwork>,
    pub mode: EvaluationMode,
}

impl Evaluator {
    pub fn new() -> Self {
        Evaluator {
            nnue_network: None,
            mode: EvaluationMode::Handcrafted,
        }
    }

    pub fn with_nnue(network: crate::nnue::NNUENetwork) -> Self {
        Evaluator {
            nnue_network: Some(network),
            mode: EvaluationMode::NNUE,
        }
    }

    pub fn set_mode(&mut self, mode: EvaluationMode) {
        self.mode = mode;
    }

    pub fn evaluate(&self, board: &Board) -> Score {
        self.evaluate_with_accumulator(board, None)
    }

    pub fn evaluate_with_accumulator(
        &self,
        board: &Board,
        acc: Option<&crate::nnue::NNUEAccumulator>,
    ) -> Score {
        match self.mode {
            EvaluationMode::NNUE => {
                if let Some(ref network) = self.nnue_network {
                    if let Some(accumulator) = acc {
                        network.evaluate_accumulator(accumulator, board.turn)
                    } else {
                        network.evaluate(board)
                    }
                } else {
                    Self::hand_crafted_evaluate(board)
                }
            }
            EvaluationMode::Hybrid => {
                let hc_score = Self::hand_crafted_evaluate(board);
                if let Some(ref network) = self.nnue_network {
                    let nn_score = if let Some(accumulator) = acc {
                        network.evaluate_accumulator(accumulator, board.turn)
                    } else {
                        network.evaluate(board)
                    };
                    (hc_score * 3 + nn_score * 7) / 10
                } else {
                    hc_score
                }
            }
            EvaluationMode::Handcrafted => Self::hand_crafted_evaluate(board),
        }
    }

    pub fn hand_crafted_evaluate(board: &Board) -> Score {
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

    pub fn load_nnue_weights(&mut self, path: &str) -> Result<(), String> {
        let weights = crate::nnue::NNUESerializer::load(path)?;
        self.nnue_network = Some(crate::nnue::NNUENetwork::from_weights(weights));
        self.mode = EvaluationMode::NNUE;
        Ok(())
    }

    pub fn enable_nnue(&mut self) {
        if self.nnue_network.is_some() {
            self.mode = EvaluationMode::NNUE;
        }
    }

    pub fn disable_nnue(&mut self) {
        self.mode = EvaluationMode::Handcrafted;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_starting_position_evaluation() {
        let evaluator = Evaluator::new();
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let score = evaluator.evaluate(&board);
        assert!(score.abs() < 10000);
    }

    #[test]
    fn test_hand_crafted_evaluation() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let score = Evaluator::hand_crafted_evaluate(&board);
        assert!(score.abs() < 10000);
    }
}
