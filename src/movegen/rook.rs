use crate::board::{Board, ChessMove};
use crate::board::pieces::{Piece, same_color};

pub fn generate(
    board: &Board,
    rank: usize,
    file: usize,
    moves: &mut Vec<ChessMove>,
) {
    let piece = board.squares[rank][file];

    let directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ];

    for (dr, df) in directions {
        let mut r = rank as isize + dr;
        let mut f = file as isize + df;

        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let target = board.squares[r as usize][f as usize];

            if target == Piece::Empty {
                moves.push(ChessMove::new(
                    rank,
                    file,
                    r as usize,
                    f as usize,
                ));
            } else {
                if !same_color(piece, target) {
                    moves.push(ChessMove::new(
                        rank,
                        file,
                        r as usize,
                        f as usize,
                    ));
                }

                break;
            }

            r += dr;
            f += df;
        }
    }
}
