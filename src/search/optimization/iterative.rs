use crate::board::Board;
use crate::board::chess_move::ChessMove;
use crate::evaluation::Score;
use super::alphabeta::AlphaBeta;
use super::aspiration::AspirationWindows;
use crate::transposition_table::TranspositionTable;
use crate::move_ordering::MoveOrderer;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct IterativeDeepening {
    pub best_move: Option<ChessMove>,
    pub best_score: Score,
    pub depth_achieved: u32,
    pub time_spent_ms: u128,
    pub ab: AlphaBeta,
    pub stop_flag: Option<Arc<AtomicBool>>,
    pub gpu_enabled: bool,
    pub gpu_batch_min: usize,
    pub gpu_max_depth: u32,
    pub gpu: Option<Arc<crate::nnue::gpu::GpuNnue>>,
    gpu_checked: bool,
}

impl IterativeDeepening {
    pub fn new() -> Self {
        IterativeDeepening {
            best_move: None,
            best_score: 0,
            depth_achieved: 0,
            time_spent_ms: 0,
            ab: AlphaBeta::new(),
            stop_flag: None,
            gpu_enabled: false,
            gpu_batch_min: 8,
            gpu_max_depth: 16,
            gpu: None,
            gpu_checked: false,
        }
    }

    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        self.gpu_enabled = enabled;
        if !enabled {
            self.gpu = None;
            self.gpu_checked = false;
        }
    }

    pub fn set_gpu_batch_min(&mut self, min: usize) {
        self.gpu_batch_min = min.max(2);
    }

    pub fn set_gpu_max_depth(&mut self, depth: u32) {
        self.gpu_max_depth = depth;
    }

    pub fn invalidate_gpu(&mut self) {
        self.gpu = None;
        self.gpu_checked = false;
    }

    fn ensure_gpu(&mut self, evaluator: &crate::evaluation::Evaluator) {
        if !self.gpu_enabled || self.gpu_checked {
            return;
        }
        self.gpu_checked = true;
        if matches!(evaluator.mode, crate::evaluation::EvaluationMode::NNUE) {
            if let Some(net) = &evaluator.nnue_network {
                self.gpu = crate::nnue::gpu::GpuNnue::from_weights(&net.weights)
                    .map(std::sync::Arc::new);
            }
        }
    }

    pub fn set_stop_flag(&mut self, flag: Arc<AtomicBool>) {
        self.stop_flag = Some(Arc::clone(&flag));
    }

    fn stop_requested(&self) -> bool {
        if let Some(ref flag) = self.stop_flag {
            return flag.load(Ordering::Relaxed);
        }
        false
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_limit_ms: Option<u128>,
    ) -> Option<ChessMove> {
        let evaluator = crate::evaluation::Evaluator::new();
        self.search_with_eval(board, max_depth, time_limit_ms, &evaluator)
    }

    pub fn search_with_eval(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_limit_ms: Option<u128>,
        evaluator: &crate::evaluation::Evaluator,
    ) -> Option<ChessMove> {
        let start = Instant::now();
        let time_limit = time_limit_ms.map(|ms| Duration::from_millis(ms as u64));

        let moves = board.generate_moves();
        if moves.is_empty() {
            return None;
        }

        let mut tt = TranspositionTable::new(16);
        let mut orderer = MoveOrderer::new(max_depth);
        self.ensure_gpu(evaluator);
        self.ab = AlphaBeta::new();
        self.ab.gpu = self.gpu.clone();
        self.ab.gpu_batch_min = self.gpu_batch_min;
        self.ab.gpu_max_depth = self.gpu_max_depth;
        if let Some(ref flag) = self.stop_flag {
            self.ab.set_stop_flag(Arc::clone(flag));
        }
        let mut aspiration = AspirationWindows::new();

        let root_accumulator = if evaluator.mode != crate::evaluation::EvaluationMode::Handcrafted {
            evaluator.nnue_network.as_ref().map(|net| net.create_accumulator(board))
        } else {
            None
        };

        self.best_move = Some(moves[0]);

        for depth in 1..=max_depth {
            if self.stop_requested() {
                break;
            }
            if let Some(limit) = time_limit {
                if start.elapsed() > limit {
                    break;
                }
            }

            self.ab.best_move_at_root = None;
            self.ab.root_depth = 0;

            let score = aspiration.search_with_eval(
                &mut self.ab,
                board,
                depth,
                self.best_score,
                &mut tt,
                &mut orderer,
                evaluator,
                root_accumulator.as_ref(),
            );
            
            if self.ab.is_aborted || self.stop_requested() {
                break;
            }

            self.best_score = score;
            self.depth_achieved = depth;

            if let Some(best) = self.ab.best_move_at_root {
                self.best_move = Some(best);
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

    pub fn search_with_node_limit(
        &mut self,
        board: &mut Board,
        max_nodes: u64,
        evaluator: &crate::evaluation::Evaluator,
    ) -> Option<ChessMove> {
        let start = Instant::now();
        let moves = board.generate_moves();
        if moves.is_empty() {
            return None;
        }

        let mut tt = TranspositionTable::new(16);
        let mut orderer = MoveOrderer::new(64);
        self.ensure_gpu(evaluator);
        self.ab = AlphaBeta::new();
        self.ab.gpu = self.gpu.clone();
        self.ab.gpu_batch_min = self.gpu_batch_min;
        self.ab.gpu_max_depth = self.gpu_max_depth;
        if let Some(ref flag) = self.stop_flag {
            self.ab.set_stop_flag(Arc::clone(flag));
        }
        self.ab.node_limit = Some(max_nodes);
        let mut aspiration = AspirationWindows::new();

        let root_accumulator = if evaluator.mode != crate::evaluation::EvaluationMode::Handcrafted {
            evaluator.nnue_network.as_ref().map(|net| net.create_accumulator(board))
        } else {
            None
        };

        self.best_move = Some(moves[0]);

        for depth in 1..=64 {
            if self.ab.total_nodes() >= max_nodes || self.ab.is_aborted || self.stop_requested() {
                break;
            }

            self.ab.best_move_at_root = None;
            self.ab.root_depth = 0;

            let score = aspiration.search_with_eval(
                &mut self.ab,
                board,
                depth,
                self.best_score,
                &mut tt,
                &mut orderer,
                evaluator,
                root_accumulator.as_ref(),
            );

            if !self.ab.is_aborted {
                self.best_score = score;
                self.depth_achieved = depth;
                if let Some(best) = self.ab.best_move_at_root {
                    self.best_move = Some(best);
                }
            } else {
                if let Some(best) = self.ab.best_move_at_root {
                    self.best_move = Some(best);
                }
                break;
            }
        }

        self.time_spent_ms = start.elapsed().as_millis();
        self.best_move
    }
}
