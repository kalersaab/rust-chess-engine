pub mod optimization;
pub mod qnode;

use crate::board::Board;
use crate::opening_book::OpeningBook;
use crate::evaluation::Score;
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
    pub gpu_batches: u64,
    pub gpu_children_total: u64,
    pub hints_used: u64,
    pub tail_gpu_batch_min: u64,
}

pub struct Searcher {
    pub stats: SearchStats,
    pub best_score: Score,
    pub tt: crate::transposition_table::TranspositionTable,
    pub move_orderer: crate::move_ordering::MoveOrderer,
    pub opening_book: OpeningBook,
    pub evaluator: crate::evaluation::Evaluator,
    id_search: IterativeDeepening,
    start_time: Option<std::time::Instant>,
    time_limit: Option<std::time::Duration>,
    stop_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            stats: SearchStats::default(),
            best_score: 0,
            tt: crate::transposition_table::TranspositionTable::new(16),
            move_orderer: crate::move_ordering::MoveOrderer::new(20),
            opening_book: OpeningBook::new(),
            evaluator: crate::evaluation::Evaluator::new(),
            id_search: IterativeDeepening::new(),
            start_time: None,
            time_limit: None,
            stop_flag: None,
        }
    }

    pub fn set_tt_size(&mut self, size_mb: u32) {
        self.tt = crate::transposition_table::TranspositionTable::new(size_mb);
    }

    pub fn set_evaluator(&mut self, evaluator: crate::evaluation::Evaluator) {
        self.evaluator = evaluator;
        self.id_search.invalidate_gpu();
    }

    pub fn set_evaluation_mode(&mut self, mode: crate::evaluation::EvaluationMode) {
        self.evaluator.set_mode(mode);
        self.id_search.invalidate_gpu();
    }

    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        self.id_search.set_gpu_enabled(enabled);
    }

    pub fn set_gpu_batch_min(&mut self, min: usize) {
        self.id_search.set_gpu_batch_min(min);
    }

    pub fn set_gpu_max_depth(&mut self, depth: u32) {
        self.id_search.set_gpu_max_depth(depth);
    }

    pub fn gpu_active(&self) -> bool {
        self.id_search.gpu.is_some()
    }

    pub fn set_stop_flag(&mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        self.stop_flag = Some(std::sync::Arc::clone(&flag));
        self.id_search.set_stop_flag(flag);
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
        self.stats.gpu_batches = self.id_search.ab.gpu_batches;
        self.stats.gpu_children_total = self.id_search.ab.gpu_children_total;
        self.stats.hints_used = self.id_search.ab.hints_used;
        self.stats.tail_gpu_batch_min = self.id_search.gpu_batch_min as u64;
        self.best_score = self.id_search.best_score;

        if let Some(start) = self.start_time {
            self.stats.time_ms = start.elapsed().as_millis();
        }

        (best_move, self.id_search.depth_achieved)
    }

    pub fn search_fixed_nodes(
        &mut self,
        board: &mut Board,
        max_nodes: u64,
    ) -> (Option<crate::board::chess_move::ChessMove>, Score) {
        if let Some(book_move) = self.opening_book.get_book_move(board) {
            return (Some(book_move), 0);
        }

        let mv = self.id_search.search_with_node_limit(board, max_nodes, &self.evaluator);
        self.stats.nodes = self.id_search.ab.nodes;
        self.stats.qnodes = self.id_search.ab.qnodes;
        self.stats.cutoffs = self.id_search.ab.cutoffs;
        self.stats.search_depth = self.id_search.depth_achieved;
        self.stats.time_ms = self.id_search.time_spent_ms;
        self.stats.gpu_batches = self.id_search.ab.gpu_batches;
        self.stats.gpu_children_total = self.id_search.ab.gpu_children_total;
        self.stats.hints_used = self.id_search.ab.hints_used;
        self.stats.tail_gpu_batch_min = self.id_search.gpu_batch_min as u64;
        self.best_score = self.id_search.best_score;

        (mv, self.id_search.best_score)
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
        self.best_score = self.id_search.best_score;

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

    #[test]
    fn test_search_fixed_nodes() {
        let mut searcher = Searcher::new();
        let mut board = Board::new();
        let (best_move, _score) = searcher.search_fixed_nodes(&mut board, 500);
        assert!(best_move.is_some());
        let total_nodes = searcher.stats.nodes + searcher.stats.qnodes;
        assert!(total_nodes >= 50, "Should search some nodes, got {}", total_nodes);
        assert!(total_nodes <= 1200, "Should terminate close to node limit, got {}", total_nodes);
    }

    #[test]
    fn test_search_nnue_gpu_and_cpu_fixed_nodes() {
        let evaluator_gpu = crate::evaluation::Evaluator::with_nnue(crate::nnue::NNUENetwork::new());
        let evaluator_cpu = crate::evaluation::Evaluator::with_nnue(crate::nnue::NNUENetwork::new());

        let mut s_gpu = Searcher::new();
        s_gpu.set_evaluator(evaluator_gpu);
        s_gpu.set_gpu_enabled(true);
        s_gpu.set_gpu_batch_min(4);
        let mut board = Board::new();
        let (mv_gpu, score_gpu) = s_gpu.search_fixed_nodes(&mut board, 2000);
        assert!(mv_gpu.is_some(), "GPU search found no move");
        assert!(score_gpu.abs() < 32000);

        let mut s_cpu = Searcher::new();
        s_cpu.set_evaluator(evaluator_cpu);
        s_cpu.set_gpu_enabled(false);
        let mut board = Board::new();
        let (mv_cpu, score_cpu) = s_cpu.search_fixed_nodes(&mut board, 2000);
        assert!(mv_cpu.is_some(), "CPU search found no move");
        assert!(score_cpu.abs() < 32000);

        assert!(s_gpu.stats.nodes + s_gpu.stats.qnodes >= 500);
        assert!(s_cpu.stats.nodes + s_cpu.stats.qnodes >= 500);

        eprintln!(
            "nnue fixnodes: gpu={:?} @ {} cp ({}n), cpu={:?} @ {} cp ({}n)",
            mv_gpu,
            score_gpu,
            s_gpu.stats.nodes + s_gpu.stats.qnodes,
            mv_cpu,
            score_cpu,
            s_cpu.stats.nodes + s_cpu.stats.qnodes
        );
    }
}
