pub mod optimization;

use crate::board::Board;
use optimization::alphabeta::AlphaBeta;
use optimization::iterative::IterativeDeepening;

#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub nodes: u64,
    pub qnodes: u64,
    pub cutoffs: u64,
    pub tt_hits: u64,
    pub tt_misses: u64,
    pub killer_moves_used: u64,
    pub search_depth: u32,
    pub time_ms: u128,
}

pub struct Searcher {
    pub stats: SearchStats,
    pub tt: crate::transposition_table::TranspositionTable,
    pub move_orderer: crate::move_ordering::MoveOrderer,
    ab_search: AlphaBeta,
    id_search: IterativeDeepening,
    start_time: Option<std::time::Instant>,
    time_limit: Option<std::time::Duration>,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            stats: SearchStats::default(),
            tt: crate::transposition_table::TranspositionTable::new(16),
            move_orderer: crate::move_ordering::MoveOrderer::new(20),
            ab_search: AlphaBeta::new(),
            id_search: IterativeDeepening::new(),
            start_time: None,
            time_limit: None,
        }
    }

    pub fn set_tt_size(&mut self, size_mb: u32) {
        self.tt = crate::transposition_table::TranspositionTable::new(size_mb);
    }

    pub fn search_with_time_management(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_available_ms: u128,
    ) -> (Option<crate::board::chess_move::ChessMove>, u32) {
        self.time_limit = Some(std::time::Duration::from_millis(time_available_ms as u64));
        self.start_time = Some(std::time::Instant::now());
        
        let best_move = self.id_search.search(board, max_depth, Some(time_available_ms));
        
        self.stats.search_depth = self.id_search.depth_achieved;
        self.stats.nodes = self.ab_search.nodes;
        self.stats.qnodes = self.ab_search.qnodes;
        self.stats.cutoffs = self.ab_search.cutoffs;

        if let Some(start) = self.start_time {
            self.stats.time_ms = start.elapsed().as_millis();
        }

        (best_move, self.id_search.depth_achieved)
    }

    pub fn find_best_move(&mut self, board: &mut Board, _depth: u32) -> Option<crate::board::chess_move::ChessMove> {
        let moves = board.generate_moves();
        moves.first().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_searcher_creation() {
        let searcher = Searcher::new();
        assert_eq!(searcher.stats.nodes, 0);
    }
}
