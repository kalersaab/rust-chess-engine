use crate::board::{Board, ChessMove};
use crate::board::pieces::{Piece, same_color};

pub fn generate(
    board: &Board,
    rank: usize,
    file: usize,
    moves: &mut Vec<ChessMove>,
) {
    let piece = board.squares[rank][file];

    let offsets = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ];

    for (dr, df) in offsets {
        let target_rank = rank as isize + dr;
        let target_file = file as isize + df;

        if target_rank < 0
            || target_rank >= 8
            || target_file < 0
            || target_file >= 8
        {
            continue;
        }

        let target = board.squares[target_rank as usize][target_file as usize];

        if target == Piece::Empty || !same_color(piece, target) {
            moves.push(ChessMove::new(
                rank,
                file,
                target_rank as usize,
                target_file as usize,
            ));
        }
    }
}
