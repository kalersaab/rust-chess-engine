use crate::board::{Board, ChessMove, Color};
use crate::board::pieces::{Piece, same_color};

pub fn generate(
    board: &Board,
    rank: usize,
    file: usize,
    moves: &mut Vec<ChessMove>,
) {
    let piece = board.squares[rank][file];

    for dr in -1isize..=1 {
        for df in -1isize..=1 {
            if dr == 0 && df == 0 {
                continue;
            }

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

    match piece {
        Piece::WhiteKing => {
            if board.castling_rights.white_kingside
                && rank == 7
                && file == 4
                && board.squares[7][5] == Piece::Empty
                && board.squares[7][6] == Piece::Empty
                && board.squares[7][7] == Piece::WhiteRook
                && !board.is_in_check(Color::White)
                && !board.is_square_attacked((7, 4), Color::Black)
                && !board.is_square_attacked((7, 5), Color::Black)
            {
                moves.push(ChessMove::castling(rank, file, rank, 6));
            }

            if board.castling_rights.white_queenside
                && rank == 7
                && file == 4
                && board.squares[7][3] == Piece::Empty
                && board.squares[7][2] == Piece::Empty
                && board.squares[7][1] == Piece::Empty
                && board.squares[7][0] == Piece::WhiteRook
                && !board.is_in_check(Color::White)
                && !board.is_square_attacked((7, 4), Color::Black)
                && !board.is_square_attacked((7, 3), Color::Black)
            {
                moves.push(ChessMove::castling(rank, file, rank, 2));
            }
        }

        Piece::BlackKing => {
            if board.castling_rights.black_kingside
                && rank == 0
                && file == 4
                && board.squares[0][5] == Piece::Empty
                && board.squares[0][6] == Piece::Empty
                && board.squares[0][7] == Piece::BlackRook
                && !board.is_in_check(Color::Black)
                && !board.is_square_attacked((0, 4), Color::White)
                && !board.is_square_attacked((0, 5), Color::White)
            {
                moves.push(ChessMove::castling(rank, file, rank, 6));
            }

            if board.castling_rights.black_queenside
                && rank == 0
                && file == 4
                && board.squares[0][3] == Piece::Empty
                && board.squares[0][2] == Piece::Empty
                && board.squares[0][1] == Piece::Empty
                && board.squares[0][0] == Piece::BlackRook
                && !board.is_in_check(Color::Black)
                && !board.is_square_attacked((0, 4), Color::White)
                && !board.is_square_attacked((0, 3), Color::White)
            {
                moves.push(ChessMove::castling(rank, file, rank, 2));
            }
        }

        _ => {}
    }
}
