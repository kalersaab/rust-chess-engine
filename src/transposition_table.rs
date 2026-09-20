use crate::board::Board;
use crate::evaluation::Score;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundType {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
pub struct TranspositionEntry {
    pub hash: u64,
    pub depth: u32,
    pub score: Score,
    pub bound: BoundType,
    pub age: u32,
}

pub struct TranspositionTable {
    entries: HashMap<u64, TranspositionEntry>,
    age: u32,
    max_entries: usize,
    hits: u64,
    misses: u64,
    collisions: u64,
}

impl TranspositionTable {
    pub fn new(size_mb: u32) -> Self {
        let max_entries = (size_mb as usize * 1024 * 1024) / 32;
        TranspositionTable {
            entries: HashMap::with_capacity(max_entries),
            age: 0,
            max_entries,
            hits: 0,
            misses: 0,
            collisions: 0,
        }
    }

    pub fn store(
        &mut self,
        board: &Board,
        depth: u32,
        score: Score,
        bound: BoundType,
    ) {
        let hash = Self::zobrist_hash(board);

        if let Some(existing) = self.entries.get(&hash) {
            if depth < existing.depth && existing.age == self.age {
                return;
            }
            self.collisions += 1;
        }

        let entry = TranspositionEntry {
            hash,
            depth,
            score,
            bound,
            age: self.age,
        };

        if self.entries.len() >= self.max_entries {
            self.clear();
        }

        self.entries.insert(hash, entry);
    }

    pub fn lookup(&mut self, board: &Board, depth: u32) -> Option<TranspositionEntry> {
        let hash = Self::zobrist_hash(board);

        if let Some(entry) = self.entries.get(&hash) {
            if entry.depth >= depth {
                self.hits += 1;
                return Some(*entry);
            }
        }

        self.misses += 1;
        None
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.age = self.age.wrapping_add(1);
    }

    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.collisions = 0;
    }

    pub fn stats(&self) -> (u64, u64, u64, usize) {
        (self.hits, self.misses, self.collisions, self.entries.len())
    }

    fn zobrist_hash(board: &Board) -> u64 {
        let mut hash: u64 = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                if piece != crate::board::pieces::Piece::Empty {
                    let piece_index = Self::piece_to_zobrist_index(piece);
                    hash ^= Self::zobrist_table(piece_index, rank, file);
                }
            }
        }

        if board.turn == crate::board::Color::Black {
            hash ^= Self::zobrist_side_to_move();
        }

        let castling_bits = Self::castling_to_bits(&board.castling_rights);
        for i in 0..4 {
            if castling_bits & (1 << i) != 0 {
                hash ^= Self::zobrist_castling(i);
            }
        }

        if let Some((_ep_rank, ep_file)) = board.en_passant_square {
            hash ^= Self::zobrist_en_passant(ep_file);
        }

        hash
    }

    fn piece_to_zobrist_index(piece: crate::board::pieces::Piece) -> usize {
        use crate::board::pieces::Piece;
        match piece {
            Piece::WhitePawn => 0,
            Piece::WhiteKnight => 1,
            Piece::WhiteBishop => 2,
            Piece::WhiteRook => 3,
            Piece::WhiteQueen => 4,
            Piece::WhiteKing => 5,
            Piece::BlackPawn => 6,
            Piece::BlackKnight => 7,
            Piece::BlackBishop => 8,
            Piece::BlackRook => 9,
            Piece::BlackQueen => 10,
            Piece::BlackKing => 11,
            Piece::Empty => 12,
        }
    }

    fn castling_to_bits(cr: &crate::board::CastlingRights) -> u8 {
        let mut bits = 0u8;
        if cr.white_kingside {
            bits |= 1;
        }
        if cr.white_queenside {
            bits |= 2;
        }
        if cr.black_kingside {
            bits |= 4;
        }
        if cr.black_queenside {
            bits |= 8;
        }
        bits
    }

    fn zobrist_table(piece: usize, rank: usize, file: usize) -> u64 {
        let mut hash = 0u64;
        hash = hash.wrapping_mul(65599).wrapping_add(piece as u64);
        hash = hash.wrapping_mul(65599).wrapping_add(rank as u64);
        hash = hash.wrapping_mul(65599).wrapping_add(file as u64);
        hash = hash.wrapping_add(0x9e3779b97f4a7c15); // Random constant
        hash ^ (hash >> 33)
    }

    fn zobrist_side_to_move() -> u64 {
        0x123456789abcdef0u64
    }

    fn zobrist_castling(castling_index: u8) -> u64 {
        let bases = [
            0x8f6b40f8d1b2e9c4u64,
            0xa7c5e1f3b9d8f2e1u64,
            0xc3e7a1f5b1d9e8f0u64,
            0xd1f5b7c3e9a8f2e1u64,
        ];
        bases[castling_index as usize % 4]
    }

    fn zobrist_en_passant(file: usize) -> u64 {
        let bases = [
            0x1a2b3c4d5e6f7a8bu64,
            0x2b3c4d5e6f7a8b9cu64,
            0x3c4d5e6f7a8b9ca0u64,
            0x4d5e6f7a8b9ca1b1u64,
            0x5e6f7a8b9ca1b2c2u64,
            0x6f7a8b9ca1b2c3d3u64,
            0x7a8b9ca1b2c3d4e4u64,
            0x8b9ca1b2c3d4e5f5u64,
        ];
        bases[file % 8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_store_and_lookup() {
        let mut tt = TranspositionTable::new(16);
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");

        tt.store(&board, 3, 0, BoundType::Exact);

        let result = tt.lookup(&mut board.clone(), 3);
        assert!(result.is_some());
        assert_eq!(result.unwrap().bound, BoundType::Exact);
        assert_eq!(result.unwrap().depth, 3);
    }

    #[test]
    fn test_tt_depth_cutoff() {
        let mut tt = TranspositionTable::new(16);
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");

        tt.store(&board, 3, 50, BoundType::Exact);

        assert!(tt.lookup(&mut board.clone(), 3).is_some());
        assert!(tt.lookup(&mut board.clone(), 2).is_some());

        tt.reset_stats();
        let result = tt.lookup(&mut board.clone(), 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_tt_zobrist_consistent() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");

        let hash1 = TranspositionTable::zobrist_hash(&board);
        let hash2 = TranspositionTable::zobrist_hash(&board);

        assert_eq!(hash1, hash2, "Zobrist hash should be consistent");
    }

    #[test]
    fn test_tt_different_positions_different_hashes() {
        let board1 = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let board2 =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .expect("Failed to parse FEN");

        let hash1 = TranspositionTable::zobrist_hash(&board1);
        let hash2 = TranspositionTable::zobrist_hash(&board2);

        assert_ne!(hash1, hash2, "Different positions should have different hashes");
    }
}
