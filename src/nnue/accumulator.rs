use ndarray::Array1;
use crate::board::{Board, Color};
use crate::board::chess_move::{ChessMove, MoveType};
use crate::board::pieces::Piece;
use crate::evaluation::Score;
use super::architecture::{HIDDEN_SIZE, NNUEWeights, output_scaling, relu};
use super::features::FeatureGenerator;

#[derive(Clone, Debug)]
pub struct NNUEAccumulator {
    pub values: Array1<f32>,
}

impl NNUEAccumulator {
    pub fn new() -> Self {
        NNUEAccumulator {
            values: Array1::zeros(HIDDEN_SIZE),
        }
    }

    /// Compute full accumulator values from scratch for a board position.
    pub fn compute_from_board(board: &Board, weights: &NNUEWeights) -> Self {
        let mut acc = weights.input_bias.clone();

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                if piece != Piece::Empty {
                    let sq = rank * 8 + file;
                    let feature_idx = FeatureGenerator::piece_to_feature(piece, sq);
                    let col = weights.input_weights.column(feature_idx);
                    acc = acc + &col;
                }
            }
        }

        NNUEAccumulator { values: acc }
    }

    /// Update accumulator incrementally given a move from a previous board state.
    pub fn update_move(
        &self,
        board_before: &Board,
        chess_move: &ChessMove,
        weights: &NNUEWeights,
    ) -> Self {
        let mut new_acc = self.values.clone();

        let (from_r, from_f) = chess_move.from;
        let (to_r, to_f) = chess_move.to;
        let moved_piece = board_before.squares[from_r][from_f];

        // 1. Remove moving piece from origin square
        if moved_piece != Piece::Empty {
            let from_sq = from_r * 8 + from_f;
            let feat = FeatureGenerator::piece_to_feature(moved_piece, from_sq);
            new_acc = new_acc - &weights.input_weights.column(feat);
        }

        // 2. Remove captured piece on destination square (if any)
        let captured_piece = board_before.squares[to_r][to_f];
        if captured_piece != Piece::Empty {
            let to_sq = to_r * 8 + to_f;
            let feat = FeatureGenerator::piece_to_feature(captured_piece, to_sq);
            new_acc = new_acc - &weights.input_weights.column(feat);
        }

        // 3. Handle move types
        match chess_move.move_type {
            MoveType::Castling => {
                // Place king on destination
                let to_sq = to_r * 8 + to_f;
                let king_feat = FeatureGenerator::piece_to_feature(moved_piece, to_sq);
                new_acc = new_acc + &weights.input_weights.column(king_feat);

                // Move rook
                match board_before.turn {
                    Color::White => {
                        if to_f == 6 {
                            // Kingside: rook 7,7 -> 7,5
                            let rook = board_before.squares[7][7];
                            let old_sq = 7 * 8 + 7;
                            let new_sq = 7 * 8 + 5;
                            new_acc = new_acc - &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, old_sq));
                            new_acc = new_acc + &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, new_sq));
                        } else if to_f == 2 {
                            // Queenside: rook 7,0 -> 7,3
                            let rook = board_before.squares[7][0];
                            let old_sq = 7 * 8 + 0;
                            let new_sq = 7 * 8 + 3;
                            new_acc = new_acc - &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, old_sq));
                            new_acc = new_acc + &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, new_sq));
                        }
                    }
                    Color::Black => {
                        if to_f == 6 {
                            // Kingside: rook 0,7 -> 0,5
                            let rook = board_before.squares[0][7];
                            let old_sq = 0 * 8 + 7;
                            let new_sq = 0 * 8 + 5;
                            new_acc = new_acc - &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, old_sq));
                            new_acc = new_acc + &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, new_sq));
                        } else if to_f == 2 {
                            // Queenside: rook 0,0 -> 0,3
                            let rook = board_before.squares[0][0];
                            let old_sq = 0 * 8 + 0;
                            let new_sq = 0 * 8 + 3;
                            new_acc = new_acc - &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, old_sq));
                            new_acc = new_acc + &weights.input_weights.column(FeatureGenerator::piece_to_feature(rook, new_sq));
                        }
                    }
                }
            }
            MoveType::EnPassant => {
                // Place pawn on destination
                let to_sq = to_r * 8 + to_f;
                let pawn_feat = FeatureGenerator::piece_to_feature(moved_piece, to_sq);
                new_acc = new_acc + &weights.input_weights.column(pawn_feat);

                // Remove captured pawn behind destination square
                match moved_piece {
                    Piece::WhitePawn => {
                        let cap_r = to_r + 1;
                        let cap_sq = cap_r * 8 + to_f;
                        let cap_piece = board_before.squares[cap_r][to_f];
                        if cap_piece != Piece::Empty {
                            let feat = FeatureGenerator::piece_to_feature(cap_piece, cap_sq);
                            new_acc = new_acc - &weights.input_weights.column(feat);
                        }
                    }
                    Piece::BlackPawn => {
                        let cap_r = to_r - 1;
                        let cap_sq = cap_r * 8 + to_f;
                        let cap_piece = board_before.squares[cap_r][to_f];
                        if cap_piece != Piece::Empty {
                            let feat = FeatureGenerator::piece_to_feature(cap_piece, cap_sq);
                            new_acc = new_acc - &weights.input_weights.column(feat);
                        }
                    }
                    _ => {}
                }
            }
            MoveType::Promotion(promo_piece) => {
                // Place promoted piece on destination
                let to_sq = to_r * 8 + to_f;
                let feat = FeatureGenerator::piece_to_feature(promo_piece, to_sq);
                new_acc = new_acc + &weights.input_weights.column(feat);
            }
            MoveType::Normal => {
                // Place moved piece on destination
                let to_sq = to_r * 8 + to_f;
                let feat = FeatureGenerator::piece_to_feature(moved_piece, to_sq);
                new_acc = new_acc + &weights.input_weights.column(feat);
            }
        }

        NNUEAccumulator { values: new_acc }
    }

    /// Evaluate position score directly from the accumulator in O(H) time.
    pub fn evaluate(&self, weights: &NNUEWeights, turn: Color) -> Score {
        let activated = self.values.mapv(relu);
        let raw_output = weights.output_weights.dot(&activated)[0] + weights.output_bias;
        let scaled = output_scaling(raw_output);

        if turn == Color::White {
            scaled
        } else {
            -scaled
        }
    }
}

impl Default for NNUEAccumulator {
    fn default() -> Self {
        Self::new()
    }
}
