use crate::board::Board;
use crate::board::chess_move::ChessMove;
use crate::evaluation::Score;
use super::alphabeta::AlphaBeta;
use crate::transposition_table::TranspositionTable;
use crate::move_ordering::MoveOrderer;
use std::time::{Instant, Duration};

pub struct IterativeDeepening {
    pub best_move: Option<ChessMove>,
    pub best_score: Score,
    pub depth_achieved: u32,
    pub time_spent_ms: u128,
}

impl IterativeDeepening {
    pub fn new() -> Self {
        IterativeDeepening {
            best_move: None,
            best_score: 0,
            depth_achieved: 0,
            time_spent_ms: 0,
        }
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_limit_ms: Option<u128>,
    ) -> Option<ChessMove> {
        let start = Instant::now();
        let time_limit = time_limit_ms.map(|ms| Duration::from_millis(ms as u64));

        let moves = board.generate_moves();
        if moves.is_empty() {
            return None;
        }

        let mut tt = TranspositionTable::new(16);
        let mut orderer = MoveOrderer::new(max_depth);
        let mut ab = AlphaBeta::new();

        self.best_move = Some(moves[0]);

        for depth in 1..=max_depth {
            if let Some(limit) = time_limit {
                if start.elapsed() > limit {
                    break;
                }
            }

            let score = ab.search(board, depth, -200000, 200000, &mut tt, &mut orderer);
            
            self.best_score = score;
            self.depth_achieved = depth;

            if moves.len() > 0 {
                self.best_move = Some(moves[0]);
            }

            if let Some(limit) = time_limit {
                if start.elapsed() > limit {
                    break;
                }
            }
        }

        self.time_spent_ms = start.elapsed().as_millis();
        self.best_move
    }
}
