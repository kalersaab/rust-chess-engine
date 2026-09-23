use crate::board::{Board, ChessMove};
use crate::board::pieces::Piece;
use crate::evaluation::{Evaluator, Score};
use crate::nnue::NNUEAccumulator;
use crate::transposition_table::TranspositionTable;

#[derive(Debug, Clone)]
pub struct QNode {
    pub depth: i32,
    pub alpha: Score,
    pub beta: Score,
    pub ply: u32,
}

impl QNode {
    pub fn new(alpha: Score, beta: Score, ply: u32) -> Self {
        QNode {
            depth: 0,
            alpha,
            beta,
            ply,
        }
    }

    pub fn search(
        &mut self,
        board: &Board,
        evaluator: &Evaluator,
        accumulator: Option<&NNUEAccumulator>,
        tt: &mut TranspositionTable,
        nodes: &mut u64,
    ) -> Score {
        *nodes += 1;

        if self.ply >= 64 {
            return evaluator.evaluate_with_accumulator(board, accumulator);
        }

        let static_eval = evaluator.evaluate_with_accumulator(board, accumulator);
        
        if static_eval >= self.beta {
            return self.beta;
        }

        let mut alpha = self.alpha;
        if static_eval > alpha {
            alpha = static_eval;
        }

        let delta_margin = 200;
        if static_eval + delta_margin < alpha && !board.is_in_check(board.turn) {
            return alpha;
        }

        let moves = self.generate_capture_moves(board);
        
        if moves.is_empty() {
            return static_eval;
        }

        let mut best_score = static_eval;

        for mv in moves {
            let see_score = self.static_exchange_evaluation(board, &mv);
            if see_score < 0 {
                continue;
            }

            let mut child_board = board.clone();
            if child_board.execute_move(mv.from, mv.to, mv.move_type).is_err() {
                continue;
            }

            let next_acc = match (accumulator, evaluator.nnue_network.as_ref()) {
                (Some(acc), Some(net)) => Some(acc.update_move(board, &mv, &net.weights)),
                _ => None,
            };

            let mut child_node = QNode::new(-self.beta, -alpha, self.ply + 1);
            let score = -child_node.search(
                &child_board,
                evaluator,
                next_acc.as_ref(),
                tt,
                nodes,
            );

            best_score = best_score.max(score);
            alpha = alpha.max(score);

            if alpha >= self.beta {
                break;
            }
        }

        best_score
    }

    fn generate_capture_moves(&self, board: &Board) -> Vec<ChessMove> {
        let all_moves = board.generate_moves();
        
        let mut captures: Vec<ChessMove> = all_moves
            .into_iter()
            .filter(|mv| {
                let target = board.squares[mv.to.0][mv.to.1];
                target != Piece::Empty
            })
            .collect();

        self.order_captures(board, &mut captures);
        
        captures
    }

    fn order_captures(&self, board: &Board, moves: &mut Vec<ChessMove>) {
        moves.sort_by_cached_key(|mv| {
            let attacker = board.squares[mv.from.0][mv.from.1];
            let victim = board.squares[mv.to.0][mv.to.1];
            
            let victim_value = Self::piece_value(victim);
            let attacker_value = Self::piece_value(attacker);
            
            std::cmp::Reverse(victim_value - attacker_value / 10)
        });
    }

    fn piece_value(piece: Piece) -> i32 {
        match piece {
            Piece::WhitePawn | Piece::BlackPawn => 100,
            Piece::WhiteKnight | Piece::BlackKnight => 320,
            Piece::WhiteBishop | Piece::BlackBishop => 330,
            Piece::WhiteRook | Piece::BlackRook => 500,
            Piece::WhiteQueen | Piece::BlackQueen => 900,
            Piece::WhiteKing | Piece::BlackKing => 20000,
            Piece::Empty => 0,
        }
    }

    fn static_exchange_evaluation(&self, board: &Board, mv: &ChessMove) -> Score {
        let attacker = board.squares[mv.from.0][mv.from.1];
        let victim = board.squares[mv.to.0][mv.to.1];
        
        if victim == Piece::Empty {
            return 0;
        }

        let gain = Self::piece_value(victim);
        let loss = Self::piece_value(attacker);

        if gain >= loss {
            return gain;
        }

        gain - loss
    }
}

pub struct QSearch {
    _max_ply: u32,
}

impl QSearch {
    pub fn new() -> Self {
        QSearch { _max_ply: 64 }
    }

    pub fn search(
        &self,
        board: &Board,
        alpha: Score,
        beta: Score,
        evaluator: &Evaluator,
        accumulator: Option<&NNUEAccumulator>,
        tt: &mut TranspositionTable,
        nodes: &mut u64,
    ) -> Score {
        let mut qnode = QNode::new(alpha, beta, 0);
        qnode.search(board, evaluator, accumulator, tt, nodes)
    }
}

impl Default for QSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qnode_creation() {
        let qnode = QNode::new(-1000, 1000, 0);
        assert_eq!(qnode.alpha, -1000);
        assert_eq!(qnode.beta, 1000);
        assert_eq!(qnode.ply, 0);
    }

    #[test]
    fn test_qsearch_starting_position() {
        let board = Board::new();
        let evaluator = Evaluator::new();
        let mut tt = TranspositionTable::new(1);
        let mut nodes = 0;
        
        let qsearch = QSearch::new();
        let score = qsearch.search(&board, -10000, 10000, &evaluator, None, &mut tt, &mut nodes);
        
        assert!(score.abs() < 10000);
        assert!(nodes > 0);
    }

    #[test]
    fn test_qnode_capture_generation() {
        let board = Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
            .expect("Valid FEN");
        
        let qnode = QNode::new(-1000, 1000, 0);
        let captures = qnode.generate_capture_moves(&board);
        
        assert!(captures.is_empty());
    }

    #[test]
    fn test_qnode_with_captures() {
        let board = Board::from_fen("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2")
            .expect("Valid FEN");
        
        let qnode = QNode::new(-1000, 1000, 0);
        let captures = qnode.generate_capture_moves(&board);
        
        assert!(!captures.is_empty(), "Should have at least one capture (exd4)");
    }

    #[test]
    fn test_piece_values() {
        assert_eq!(QNode::piece_value(Piece::WhitePawn), 100);
        assert_eq!(QNode::piece_value(Piece::WhiteKnight), 320);
        assert_eq!(QNode::piece_value(Piece::WhiteBishop), 330);
        assert_eq!(QNode::piece_value(Piece::WhiteRook), 500);
        assert_eq!(QNode::piece_value(Piece::WhiteQueen), 900);
        assert_eq!(QNode::piece_value(Piece::WhiteKing), 20000);
    }
}
