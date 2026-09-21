use ndarray::Array1;
use crate::board::Board;
use crate::board::pieces::Piece;
use super::architecture::INPUT_SIZE;

pub struct FeatureGenerator;

impl FeatureGenerator {
    pub fn board_to_features(board: &Board) -> Array1<f32> {
        let mut features = Array1::zeros(INPUT_SIZE);

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                let square_index = rank * 8 + file;

                if piece == Piece::Empty {
                    continue;
                }

                let feature_index = Self::piece_to_feature(piece, square_index);
                features[feature_index] = 1.0;
            }
        }

        features
    }

    pub fn active_feature_indices(board: &Board) -> Vec<usize> {
        let mut active = Vec::with_capacity(32);
        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                if piece != Piece::Empty {
                    let square_index = rank * 8 + file;
                    active.push(Self::piece_to_feature(piece, square_index));
                }
            }
        }
        active
    }

    pub fn board_to_features_incremental(
        board: &Board,
        from: (usize, usize),
        to: (usize, usize),
        prev_features: &Array1<f32>,
    ) -> Array1<f32> {
        let mut features = prev_features.clone();

        let from_piece = board.squares[from.0][from.1];
        let to_piece = board.squares[to.0][to.1];

        if from_piece != Piece::Empty {
            let feature_idx = Self::piece_to_feature(from_piece, from.0 * 8 + from.1);
            features[feature_idx] = 0.0;
        }

        if to_piece != Piece::Empty {
            let feature_idx = Self::piece_to_feature(to_piece, to.0 * 8 + to.1);
            features[feature_idx] = 0.0;
        }

        let moved_piece = board.squares[to.0][to.1];
        if moved_piece != Piece::Empty {
            let feature_idx = Self::piece_to_feature(moved_piece, to.0 * 8 + to.1);
            features[feature_idx] = 1.0;
        }

        features
    }

    pub fn piece_to_feature(piece: Piece, square_index: usize) -> usize {
        let piece_type = Self::piece_type_index(piece);
        let is_white = Self::piece_is_white(piece);

        if is_white {
            // White pieces: indices 0..383  (6 types × 64 squares)
            piece_type * 64 + square_index
        } else {
            // Black pieces: indices 384..767 (offset by 6 × 64 = 384)
            384 + piece_type * 64 + (63 - square_index)
        }
    }

    pub fn piece_type_index(piece: Piece) -> usize {
        match piece {
            Piece::WhitePawn | Piece::BlackPawn => 0,
            Piece::WhiteKnight | Piece::BlackKnight => 1,
            Piece::WhiteBishop | Piece::BlackBishop => 2,
            Piece::WhiteRook | Piece::BlackRook => 3,
            Piece::WhiteQueen | Piece::BlackQueen => 4,
            Piece::WhiteKing | Piece::BlackKing => 5,
            Piece::Empty => 12,
        }
    }

    pub fn piece_is_white(piece: Piece) -> bool {
        matches!(
            piece,
            Piece::WhitePawn
                | Piece::WhiteKnight
                | Piece::WhiteBishop
                | Piece::WhiteRook
                | Piece::WhiteQueen
                | Piece::WhiteKing
        )
    }

    pub fn feature_index_to_square(feature_index: usize) -> (usize, usize) {
        let _piece_type = feature_index / 64;
        let square = feature_index % 64;

        let rank = square / 8;
        let file = square % 8;

        (rank, file)
    }
}
