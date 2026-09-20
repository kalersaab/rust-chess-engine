use crate::board::{Board, ChessMove};
use crate::evaluation::{Evaluator, Score};
use crate::nnue::NNUEAccumulator;
use crate::transposition_table::{TranspositionTable, BoundType};
use crate::move_ordering::MoveOrderer;

pub struct AlphaBeta {
    pub nodes: u64,
    pub qnodes: u64,
    pub cutoffs: u64,
    pub best_move_at_root: Option<ChessMove>,
    pub root_depth: u32,
}

impl AlphaBeta {
    pub fn new() -> Self {
        AlphaBeta {
            nodes: 0,
            qnodes: 0,
            cutoffs: 0,
            best_move_at_root: None,
            root_depth: 0,
        }
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        depth: u32,
        alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
    ) -> Score {
        let evaluator = Evaluator::new();
        self.search_with_eval(board, depth, alpha, beta, tt, orderer, &evaluator, None)
    }

    pub fn search_with_eval(
        &mut self,
        board: &mut Board,
        depth: u32,
        mut alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
        evaluator: &Evaluator,
        accumulator: Option<&NNUEAccumulator>,
    ) -> Score {

        if depth > self.root_depth {
            self.root_depth = depth;
        }
        let is_root = depth == self.root_depth;
        if depth == 0 {
            return self.quiescence(board, alpha, beta, tt, orderer, evaluator, accumulator);
        }

        self.nodes += 1;

        let hash_move = tt.lookup(board, depth).map(|_entry| {
            ChessMove::new(0, 0, 0, 0)
        });

        let mut moves = board.generate_moves();
        
        if moves.is_empty() {
            if board.is_in_check(board.turn) {
                return -100000 + (20 - depth) as Score;
            } else {
                return 0;
            }
        }

        orderer.sort_moves(&mut moves, board, depth, hash_move);

        let mut best_score = -200000;

        for mv in moves {
            let mut next_board = board.clone();
            if next_board.execute_move(mv.from, mv.to, mv.move_type).is_err() {
                continue;
            }

            let next_acc = match (accumulator, evaluator.nnue_network.as_ref()) {
                (Some(acc), Some(net)) => Some(acc.update_move(board, &mv, &net.weights)),
                _ => None,
            };

            let score = -self.search_with_eval(
                &mut next_board,
                depth - 1,
                -beta,
                -alpha,
                tt,
                orderer,
                evaluator,
                next_acc.as_ref(),
            );

            best_score = best_score.max(score);

            if score > alpha {
                alpha = score;
                if is_root {
                    self.best_move_at_root = Some(mv);
                }
            }

            if alpha >= beta {
                self.cutoffs += 1;
                orderer.record_killer(depth, mv);
                break;
            }
        }

        tt.store(board, depth, best_score, BoundType::Exact);
        best_score
    }

    fn quiescence(
        &mut self,
        board: &Board,
        mut alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        _orderer: &MoveOrderer,
        evaluator: &Evaluator,
        accumulator: Option<&NNUEAccumulator>,
    ) -> Score {
        self.qnodes += 1;

        let static_eval = evaluator.evaluate_with_accumulator(board, accumulator);
        
        if static_eval >= beta {
            return beta;
        }

        alpha = alpha.max(static_eval);

        let moves = board.generate_moves();
        let mut best_score = static_eval;

        for mv in moves {
            let target = board.squares[mv.to.0][mv.to.1];
            if target == crate::board::pieces::Piece::Empty {
                continue;
            }

            let mut board_copy = board.clone();
            if board_copy.execute_move(mv.from, mv.to, mv.move_type).is_err() {
                continue;
            }

            let next_acc = match (accumulator, evaluator.nnue_network.as_ref()) {
                (Some(acc), Some(net)) => Some(acc.update_move(board, &mv, &net.weights)),
                _ => None,
            };

            let score = -self.quiescence(
                &board_copy,
                -beta,
                -alpha,
                tt,
                _orderer,
                evaluator,
                next_acc.as_ref(),
            );
            best_score = best_score.max(score);
            alpha = alpha.max(score);

            if alpha >= beta {
                break;
            }
        }

        best_score
    }
}
