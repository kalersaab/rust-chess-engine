use crate::board::Board;
use crate::evaluation::Score;
use crate::transposition_table::{TranspositionTable, BoundType};

pub struct TranspositionIntegration;

impl TranspositionIntegration {
    pub fn probe_hash(
        tt: &mut TranspositionTable,
        board: &Board,
        depth: u32,
        alpha: Score,
        beta: Score,
    ) -> Option<Score> {
        if let Some(entry) = tt.lookup(board, depth) {
            match entry.bound {
                BoundType::Exact => Some(entry.score),
                BoundType::Lower => {
                    if entry.score >= beta {
                        Some(entry.score)
                    } else {
                        None
                    }
                }
                BoundType::Upper => {
                    if entry.score <= alpha {
                        Some(entry.score)
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        }
    }

    pub fn store_hash(
        tt: &mut TranspositionTable,
        board: &Board,
        depth: u32,
        score: Score,
        alpha: Score,
        beta: Score,
    ) {
        let bound = if score <= alpha {
            BoundType::Upper
        } else if score >= beta {
            BoundType::Lower
        } else {
            BoundType::Exact
        };

        tt.store(board, depth, score, bound);
    }

    pub fn clear_hash(tt: &mut TranspositionTable) {
        tt.clear();
    }

    pub fn hash_stats(tt: &TranspositionTable) -> (u64, u64, u64, usize) {
        tt.stats()
    }
}
