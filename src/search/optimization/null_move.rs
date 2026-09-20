use crate::board::{Board, Color};
use crate::evaluation::Score;
use crate::transposition_table::TranspositionTable;
use crate::move_ordering::MoveOrderer;
use super::alphabeta::AlphaBeta;

pub struct NullMovePruning {
    pub min_depth: u32,
    pub reduction_factor: u32,
    pub cutoffs: u32,
}

impl NullMovePruning {
    pub fn new() -> Self {
        NullMovePruning {
            min_depth: 2,
            reduction_factor: 3,
            cutoffs: 0,
        }
    }

    pub fn can_prune(
        &self,
        board: &Board,
        depth: u32,
        beta: Score,
        static_eval: Score,
    ) -> bool {
        if depth < self.min_depth {
            return false;
        }

        if board.is_in_check(board.turn) {
            return false;
        }

        if static_eval < beta {
            return false;
        }

        true
    }

    pub fn search(
        &mut self,
        ab: &mut AlphaBeta,
        board: &mut Board,
        depth: u32,
        beta: Score,
        static_eval: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
    ) -> Option<Score> {
        if !self.can_prune(board, depth, beta, static_eval) {
            return None;
        }

        let mut null_board = board.clone();
        null_board.turn = match null_board.turn {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
        null_board.en_passant_square = None;

        let reduced_depth = ((depth - self.min_depth) / self.reduction_factor).max(1);

        let null_score = -ab.search(
            &mut null_board,
            reduced_depth,
            -beta,
            -beta + 1,
            tt,
            orderer,
        );

        if null_score >= beta {
            self.cutoffs += 1;
            Some(beta)
        } else {
            None
        }
    }
}
