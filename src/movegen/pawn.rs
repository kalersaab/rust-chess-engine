use crate::board::{Board, ChessMove};
use crate::board::pieces::{Piece, same_color};

pub fn generate(
    board: &Board,
    rank: usize,
    file: usize,
    moves: &mut Vec<ChessMove>,
) {
    let piece = board.squares[rank][file];

    let direction: isize;
    let start_rank: usize;
    let promotion_rank: usize;

    match piece {
        Piece::WhitePawn => {
            direction = -1;
            start_rank = 6;
            promotion_rank = 0;
        }

        Piece::BlackPawn => {
            direction = 1;
            start_rank = 1;
            promotion_rank = 7;
        }

        _ => return,
    }

    let next_rank = rank as isize + direction;

    if next_rank >= 0
        && next_rank < 8
        && board.squares[next_rank as usize][file] == Piece::Empty
    {
        let next_rank_usize = next_rank as usize;
        
        if next_rank_usize == promotion_rank {
            let promotion_pieces = match piece {
                Piece::WhitePawn => [Piece::WhiteQueen, Piece::WhiteRook, Piece::WhiteBishop, Piece::WhiteKnight],
                Piece::BlackPawn => [Piece::BlackQueen, Piece::BlackRook, Piece::BlackBishop, Piece::BlackKnight],
                _ => unreachable!(),
            };
            
            for promo_piece in promotion_pieces {
                moves.push(ChessMove::promotion(rank, file, next_rank_usize, file, promo_piece));
            }
        } else {
            moves.push(ChessMove::new(
                rank,
                file,
                next_rank_usize,
                file,
            ));

            if rank == start_rank {
                let double_rank = rank as isize + direction * 2;

                if board.squares[double_rank as usize][file] == Piece::Empty {
                    moves.push(ChessMove::new(
                        rank,
                        file,
                        double_rank as usize,
                        file,
                    ));
                }
            }
        }
    }

    for file_offset in [-1isize, 1isize] {
        let target_file = file as isize + file_offset;

        if next_rank < 0
            || next_rank >= 8
            || target_file < 0
            || target_file >= 8
        {
            continue;
        }

        let next_rank_usize = next_rank as usize;
        let target_file_usize = target_file as usize;
        let target = board.squares[next_rank_usize][target_file_usize];

        if target != Piece::Empty
            && same_color(piece, target) == false
        {
            if next_rank_usize == promotion_rank {
                let promotion_pieces = match piece {
                    Piece::WhitePawn => [Piece::WhiteQueen, Piece::WhiteRook, Piece::WhiteBishop, Piece::WhiteKnight],
                    Piece::BlackPawn => [Piece::BlackQueen, Piece::BlackRook, Piece::BlackBishop, Piece::BlackKnight],
                    _ => unreachable!(),
                };
                
                for promo_piece in promotion_pieces {
                    moves.push(ChessMove::promotion(rank, file, next_rank_usize, target_file_usize, promo_piece));
                }
            } else {
                moves.push(ChessMove::new(
                    rank,
                    file,
                    next_rank_usize,
                    target_file_usize,
                ));
            }
        }
    }

    for file_offset in [-1isize, 1isize] {
        let target_file = file as isize + file_offset;

        if target_file < 0 || target_file >= 8 {
            continue;
        }

        if let Some((ep_rank, ep_file)) = board.en_passant_square {
            let next_rank_usize = next_rank as usize;
            if next_rank >= 0 && next_rank < 8 && next_rank_usize == ep_rank && target_file as usize == ep_file {
                moves.push(ChessMove::en_passant(
                    rank,
                    file,
                    next_rank_usize,
                    target_file as usize,
                ));
            }
        }
    }
}
