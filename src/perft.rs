use crate::board::Board;
use crate::board::chess_move::MoveType;
use crate::board::pieces::Piece;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PerftResult {
    pub nodes: u64,
    pub captures: u64,
    pub en_passants: u64,
    pub castles: u64,
    pub promotions: u64,
    pub checks: u64,
    pub checkmates: u64,
}

impl PerftResult {
    pub fn new() -> Self {
        PerftResult {
            nodes: 0,
            captures: 0,
            en_passants: 0,
            castles: 0,
            promotions: 0,
            checks: 0,
            checkmates: 0,
        }
    }

    pub fn add(&mut self, other: &PerftResult) {
        self.nodes += other.nodes;
        self.captures += other.captures;
        self.en_passants += other.en_passants;
        self.castles += other.castles;
        self.promotions += other.promotions;
        self.checks += other.checks;
        self.checkmates += other.checkmates;
    }
}

impl Board {
    pub fn perft(&self, depth: u32) -> PerftResult {
        let mut result = PerftResult::new();
        self.perft_internal(depth, &mut result);
        result
    }

    fn perft_internal(&self, depth: u32, result: &mut PerftResult) {
        if depth == 0 {
            result.nodes += 1;
            return;
        }

        let moves = self.generate_moves();

        for mv in moves {
            match mv.move_type {
                MoveType::Castling => result.castles += 1,
                MoveType::EnPassant => result.en_passants += 1,
                MoveType::Promotion(_) => result.promotions += 1,
                MoveType::Normal => {
                    if self.squares[mv.to.0][mv.to.1] != Piece::Empty {
                        result.captures += 1;
                    }
                }
            }

            let mut next_board = self.clone();

            if next_board.execute_move(mv.from, mv.to, mv.move_type).is_ok() {
                let is_in_check = next_board.is_in_check(next_board.turn);
                
                if is_in_check {
                    let legal_moves = next_board.generate_moves();
                    if legal_moves.is_empty() {
                        result.checkmates += 1;
                    } else {
                        result.checks += 1;
                    }
                }

                next_board.perft_internal(depth - 1, result);
            }
        }
    }

    pub fn perft_divide(&self, depth: u32) -> HashMap<String, u64> {
        let mut moves_map = HashMap::new();
        let moves = self.generate_moves();

        for mv in moves {
            let mut board = self.clone();
            let from = Self::square_to_string(mv.from);
            let to = Self::square_to_string(mv.to);
            let promo_suffix = match mv.move_type {
                MoveType::Promotion(p) => match p {
                    Piece::WhiteQueen | Piece::BlackQueen => "q",
                    Piece::WhiteRook | Piece::BlackRook => "r",
                    Piece::WhiteBishop | Piece::BlackBishop => "b",
                    Piece::WhiteKnight | Piece::BlackKnight => "n",
                    _ => "",
                },
                _ => "",
            };
            let move_notation = format!("{}{}{}", from, to, promo_suffix);

            if board.execute_move(mv.from, mv.to, mv.move_type).is_ok() {
                let result = board.perft(depth - 1);
                moves_map.insert(move_notation, result.nodes);
            }
        }

        moves_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perft_starting_position_depth_1() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(1);
        assert_eq!(result.nodes, 20, "Starting position depth 1 should have 20 moves");
    }

    #[test]
    fn test_perft_starting_position_depth_2() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(2);
        assert_eq!(result.nodes, 400, "Starting position depth 2 should have 400 nodes");
    }

    #[test]
    fn test_perft_starting_position_depth_3() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(3);
        assert_eq!(result.nodes, 8_902, "Starting position depth 3 should have 8902 nodes");
    }

    #[test]
    fn test_perft_starting_position_depth_4() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(4);
        assert_eq!(result.nodes, 197_281, "Starting position depth 4 should have 197281 nodes");
    }

    #[test]
    fn test_perft_starting_position_depth_5() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(5);
        assert_eq!(result.nodes, 4_865_609, "Starting position depth 5 should have 4865609 nodes");
    }

    #[test]
    fn test_perft_kiwipete_depth_1() {
        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(1);
        assert_eq!(result.nodes, 48, "Kiwipete depth 1 should have 48 moves");
    }

    #[test]
    fn test_perft_kiwipete_depth_2() {
        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(2);
        assert_eq!(result.nodes, 2_039, "Kiwipete depth 2 should have 2039 nodes");
    }



    #[test]
    fn test_perft_kiwipete_depth_3() {
        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(3);
        assert_eq!(result.nodes, 97_862, "Kiwipete depth 3 should have 97862 nodes");
    }

    #[test]
    fn test_perft_kiwipete_depth_4() {
        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(4);
        assert_eq!(result.nodes, 4_085_603, "Kiwipete depth 4 should have 4085603 nodes");
    }
}

