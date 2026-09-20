pub mod optimization;

use crate::board::Board;
use crate::opening_book::OpeningBook;
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
    pub opening_book: OpeningBook,
    pub evaluator: crate::evaluation::Evaluator,
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
            opening_book: OpeningBook::new(),
            evaluator: crate::evaluation::Evaluator::new(),
            id_search: IterativeDeepening::new(),
            start_time: None,
            time_limit: None,
        }
    }

    pub fn set_tt_size(&mut self, size_mb: u32) {
        self.tt = crate::transposition_table::TranspositionTable::new(size_mb);
    }

    pub fn set_evaluator(&mut self, evaluator: crate::evaluation::Evaluator) {
        self.evaluator = evaluator;
    }

    pub fn set_evaluation_mode(&mut self, mode: crate::evaluation::EvaluationMode) {
        self.evaluator.set_mode(mode);
    }

    pub fn load_opening_book<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<usize, String> {
        self.opening_book.load_epd(path)
    }

    pub fn search_with_time_management(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_available_ms: u128,
    ) -> (Option<crate::board::chess_move::ChessMove>, u32) {
        self.time_limit = Some(std::time::Duration::from_millis(time_available_ms as u64));
        self.start_time = Some(std::time::Instant::now());
        
        if let Some(book_move) = self.opening_book.get_book_move(board) {
            self.stats.search_depth = 1;
            self.stats.time_ms = self.start_time.map(|s| s.elapsed().as_millis()).unwrap_or(0);
            return (Some(book_move), 1);
        }
        
        let best_move = self.id_search.search_with_eval(board, max_depth, Some(time_available_ms), &self.evaluator);
        
        self.stats.search_depth = self.id_search.depth_achieved;
        self.stats.nodes = self.id_search.ab.nodes;
        self.stats.qnodes = self.id_search.ab.qnodes;
        self.stats.cutoffs = self.id_search.ab.cutoffs;

        if let Some(start) = self.start_time {
            self.stats.time_ms = start.elapsed().as_millis();
        }

        (best_move, self.id_search.depth_achieved)
    }

    pub fn find_best_move(&mut self, board: &mut Board, depth: u32) -> Option<crate::board::chess_move::ChessMove> {
        if let Some(book_move) = self.opening_book.get_book_move(board) {
            return Some(book_move);
        }

        let result = self.id_search.search_with_eval(board, depth, None, &self.evaluator);

        self.stats.nodes = self.id_search.ab.nodes;
        self.stats.qnodes = self.id_search.ab.qnodes;
        self.stats.cutoffs = self.id_search.ab.cutoffs;
        self.stats.search_depth = self.id_search.depth_achieved;

        result
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
