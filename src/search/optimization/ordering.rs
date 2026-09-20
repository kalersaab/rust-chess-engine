use crate::board::{Board, ChessMove};
use crate::move_ordering::MoveOrderer;

pub struct MoveOrderingStrategies;

impl MoveOrderingStrategies {
    pub fn apply_mvv_lva(
        orderer: &MoveOrderer,
        moves: &mut Vec<ChessMove>,
        board: &Board,
        depth: u32,
        hash_move: Option<ChessMove>,
    ) {
        orderer.sort_moves(moves, board, depth, hash_move);
    }

    pub fn apply_killer_moves(
        orderer: &MoveOrderer,
        depth: u32,
    ) -> Vec<ChessMove> {
        orderer.get_killers(depth)
    }

    pub fn record_killer_move(
        orderer: &mut MoveOrderer,
        depth: u32,
        mv: ChessMove,
    ) {
        orderer.record_killer(depth, mv);
    }

    pub fn update_history_heuristic(
        orderer: &mut MoveOrderer,
        mv: ChessMove,
        depth: u32,
    ) {
        orderer.update_history(mv, depth);
    }

    pub fn get_history_score(
        orderer: &MoveOrderer,
        mv: ChessMove,
    ) -> u32 {
        orderer.get_history_score(mv)
    }

    pub fn clear_ordering_state(orderer: &mut MoveOrderer) {
        orderer.clear();
    }
}
