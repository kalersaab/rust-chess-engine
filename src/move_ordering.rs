use crate::board::Board;
use crate::board::chess_move::ChessMove;
use crate::board::pieces::Piece;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MoveScore(i32);

impl MoveScore {
    pub const KILLER_SCORE: i32 = 8000;
    
    pub const HASH_MOVE_SCORE: i32 = 16000;
    
    pub const MIN_CAPTURE_SCORE: i32 = 105 * 100; // pawn capturing queen = 900 centipawns
    
    pub const MAX_CAPTURE_SCORE: i32 = 900 * 100; // queen capturing pawn = 900 * 100 centipawns
}

pub struct MoveOrderer {
    killer_moves: Vec<Vec<Option<ChessMove>>>,
    history: Vec<u32>,
    max_depth: u32,
}

impl MoveOrderer {
    pub fn new(max_depth: u32) -> Self {
        MoveOrderer {
            killer_moves: vec![vec![None, None]; max_depth as usize + 1],
            history: vec![0; 12 * 64], // 12 piece types * 64 squares
            max_depth,
        }
    }

    pub fn clear(&mut self) {
        for depth_killers in &mut self.killer_moves {
            for killer in depth_killers {
                *killer = None;
            }
        }
        for h in &mut self.history {
            *h = 0;
        }
    }

    pub fn record_killer(&mut self, depth: u32, mv: ChessMove) {
        if depth as usize > self.max_depth as usize {
            return;
        }
        let idx = depth as usize;
        if self.killer_moves[idx][0] != Some(mv) {
            self.killer_moves[idx][1] = self.killer_moves[idx][0];
            self.killer_moves[idx][0] = Some(mv);
        }
    }

    pub fn update_history(&mut self, mv: ChessMove, depth: u32) {
        let piece_idx = self.piece_to_history_index(mv.from);
        let square_idx = mv.to.0 * 8 + mv.to.1;
        let idx = piece_idx * 64 + square_idx;
        
        if idx < self.history.len() {
            let bonus = (depth * depth) as u32;
            self.history[idx] = self.history[idx].saturating_add(bonus);
        }
    }

    pub fn get_killers(&self, depth: u32) -> Vec<ChessMove> {
        if depth as usize > self.max_depth as usize {
            return Vec::new();
        }
        
        self.killer_moves[depth as usize]
            .iter()
            .filter_map(|&mv| mv)
            .collect()
    }

    pub fn get_history_score(&self, mv: ChessMove) -> u32 {
        let piece_idx = self.piece_to_history_index(mv.from);
        let square_idx = mv.to.0 * 8 + mv.to.1;
        let idx = piece_idx * 64 + square_idx;
        
        if idx < self.history.len() {
            self.history[idx]
        } else {
            0
        }
    }

    fn piece_to_history_index(&self, _from: (usize, usize)) -> usize {
        0
    }

    pub fn score_move(
        &self,
        mv: ChessMove,
        board: &Board,
        depth: u32,
        hash_move: Option<ChessMove>,
    ) -> MoveScore {
        if let Some(hm) = hash_move {
            if hm.from == mv.from && hm.to == mv.to {
                return MoveScore(MoveScore::HASH_MOVE_SCORE);
            }
        }

        let target_piece = board.squares[mv.to.0][mv.to.1];
        if target_piece != Piece::Empty {
            let score = Self::mvv_lva_score(mv, target_piece, board);
            return MoveScore(score);
        }

        let killers = self.get_killers(depth);
        for killer in killers {
            if killer.from == mv.from && killer.to == mv.to {
                return MoveScore(MoveScore::KILLER_SCORE);
            }
        }

        let history_score = self.get_history_score(mv);
        MoveScore((history_score as i32).min(7999))
    }

    fn mvv_lva_score(mv: ChessMove, victim: Piece, board: &Board) -> i32 {
        let attacker = board.squares[mv.from.0][mv.from.1];
        
        let victim_value = Self::piece_value(victim);
        let attacker_value = Self::piece_value(attacker);
        
        (victim_value * 10 - attacker_value) as i32
    }

    fn piece_value(piece: Piece) -> u32 {
        use crate::board::pieces::Piece;
        match piece {
            Piece::Empty => 0,
            Piece::WhitePawn | Piece::BlackPawn => 1,
            Piece::WhiteKnight | Piece::BlackKnight => 3,
            Piece::WhiteBishop | Piece::BlackBishop => 3,
            Piece::WhiteRook | Piece::BlackRook => 5,
            Piece::WhiteQueen | Piece::BlackQueen => 9,
            Piece::WhiteKing | Piece::BlackKing => 0, // King can't be captured
        }
    }

    pub fn sort_moves(
        &self,
        moves: &mut Vec<ChessMove>,
        board: &Board,
        depth: u32,
        hash_move: Option<ChessMove>,
    ) {
        let mut scored_moves: Vec<_> = moves
            .iter()
            .map(|&mv| (mv, self.score_move(mv, board, depth, hash_move)))
            .collect();

        scored_moves.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending

        for (i, (mv, _)) in scored_moves.iter().enumerate() {
            moves[i] = *mv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_killer_move_recording() {
        let mut orderer = MoveOrderer::new(6);
        let mv = ChessMove::new(1, 1, 2, 2);

        orderer.record_killer(2, mv);
        let killers = orderer.get_killers(2);
        assert_eq!(killers.len(), 1);
        assert_eq!(killers[0], mv);

        orderer.record_killer(2, mv);
        assert_eq!(orderer.get_killers(2).len(), 1);
    }

    #[test]
    fn test_killer_move_shift() {
        let mut orderer = MoveOrderer::new(6);
        let mv1 = ChessMove::new(1, 1, 2, 2);
        let mv2 = ChessMove::new(3, 3, 4, 4);

        orderer.record_killer(2, mv1);
        orderer.record_killer(2, mv2);

        let killers = orderer.get_killers(2);
        assert_eq!(killers.len(), 2);
        assert_eq!(killers[0], mv2); // Most recent first
        assert_eq!(killers[1], mv1);
    }

    #[test]
    fn test_history_heuristic() {
        let mut orderer = MoveOrderer::new(6);
        let mv = ChessMove::new(1, 1, 2, 2);

        assert_eq!(orderer.get_history_score(mv), 0);

        orderer.update_history(mv, 3);
        let score = orderer.get_history_score(mv);
        assert_eq!(score, 9); // depth^2 = 3^2 = 9
    }

    #[test]
    fn test_mvv_lva_scoring() {
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .expect("Failed to parse FEN");

        let pawn_captures_pawn = MoveOrderer::mvv_lva_score(
            ChessMove::new(1, 0, 2, 1),
            Piece::BlackPawn,
            &board,
        );
        assert_eq!(pawn_captures_pawn, 9);

        let pawn_captures_queen = MoveOrderer::mvv_lva_score(
            ChessMove::new(1, 0, 0, 3),
            Piece::BlackQueen,
            &board,
        );
        assert_eq!(pawn_captures_queen, 89);

        let queen_captures_pawn = MoveOrderer::mvv_lva_score(
            ChessMove::new(0, 3, 1, 0),
            Piece::BlackPawn,
            &board,
        );
        assert_eq!(queen_captures_pawn, 1);
    }

    #[test]
    fn test_move_sorting() {
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .expect("Failed to parse FEN");

        let orderer = MoveOrderer::new(6);
        let mut moves = vec![
            ChessMove::new(1, 0, 2, 0), // pawn move
            ChessMove::new(0, 1, 2, 2), // knight move
            ChessMove::new(0, 0, 1, 0), // rook move (invalid but for testing)
        ];

        orderer.sort_moves(&mut moves, &board, 1, None);

        assert_eq!(moves.len(), 3);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut orderer = MoveOrderer::new(6);
        let mv = ChessMove::new(1, 1, 2, 2);

        orderer.record_killer(2, mv);
        orderer.update_history(mv, 3);

        orderer.clear();

        assert_eq!(orderer.get_killers(2).len(), 0);
        assert_eq!(orderer.get_history_score(mv), 0);
    }
}
