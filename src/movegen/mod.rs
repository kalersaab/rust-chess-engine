pub mod pawn;
pub mod knight;
pub mod bishop;
pub mod rook;
pub mod queen;
pub mod king;

use crate::board::{Board, ChessMove};

pub struct MoveGenerator;

impl MoveGenerator {
    pub fn generate_pawn_moves(
        board: &Board,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        pawn::generate(board, rank, file, moves);
    }

    pub fn generate_knight_moves(
        board: &Board,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        knight::generate(board, rank, file, moves);
    }

    pub fn generate_bishop_moves(
        board: &Board,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        bishop::generate(board, rank, file, moves);
    }

    pub fn generate_rook_moves(
        board: &Board,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        rook::generate(board, rank, file, moves);
    }

    pub fn generate_queen_moves(
        board: &Board,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        queen::generate(board, rank, file, moves);
    }

    pub fn generate_king_moves(
        board: &Board,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        king::generate(board, rank, file, moves);
    }
}
