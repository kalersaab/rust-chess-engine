use crate::board::{Board, ChessMove};
use super::{bishop, rook};

pub fn generate(
    board: &Board,
    rank: usize,
    file: usize,
    moves: &mut Vec<ChessMove>,
) {
    bishop::generate(board, rank, file, moves);
    rook::generate(board, rank, file, moves);
}
