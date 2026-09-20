use crate::board::Board;
use crate::evaluation::Score;
use crate::transposition_table::TranspositionTable;
use crate::move_ordering::MoveOrderer;
use super::alphabeta::AlphaBeta;

pub struct AspirationWindows {
    pub window_size: Score,
    pub re_searches: u32,
}

impl AspirationWindows {
    pub fn new() -> Self {
        AspirationWindows {
            window_size: 50,
            re_searches: 0,
        }
    }

    pub fn search(
        &mut self,
        ab: &mut AlphaBeta,
        board: &mut Board,
        depth: u32,
        prev_score: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
    ) -> Score {
        let evaluator = crate::evaluation::Evaluator::new();
        self.search_with_eval(ab, board, depth, prev_score, tt, orderer, &evaluator, None)
    }

    pub fn search_with_eval(
        &mut self,
        ab: &mut AlphaBeta,
        board: &mut Board,
        depth: u32,
        prev_score: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
        evaluator: &crate::evaluation::Evaluator,
        accumulator: Option<&crate::nnue::NNUEAccumulator>,
    ) -> Score {
        self.re_searches = 0;

        if depth < 4 {
            return ab.search_with_eval(board, depth, -200000, 200000, tt, orderer, evaluator, accumulator);
        }

        let mut alpha = prev_score - self.window_size;
        let mut beta = prev_score + self.window_size;

        loop {
            let score = ab.search_with_eval(board, depth, alpha, beta, tt, orderer, evaluator, accumulator);

            if score <= alpha {
                alpha = score - (self.window_size * 2);
                self.re_searches += 1;
                continue;
            }

            if score >= beta {
                beta = score + (self.window_size * 2);
                self.re_searches += 1;
                continue;
            }

            return score;
        }
    }
}
