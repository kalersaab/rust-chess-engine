#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChessMove {
    pub from: (usize, usize),
    pub to: (usize, usize),
}

impl ChessMove {
    pub fn new(
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
    ) -> Self {
        Self {
            from: (from_rank, from_file),
            to: (to_rank, to_file),
        }
    }
    
}