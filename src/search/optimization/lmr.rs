use crate::evaluation::Score;

pub struct LateMoveReductions {
    pub reduction_table: Vec<Vec<u32>>,
}

impl LateMoveReductions {
    pub fn new() -> Self {
        let mut lmr = LateMoveReductions {
            reduction_table: vec![vec![0; 64]; 64],
        };
        lmr.init_reductions();
        lmr
    }

    fn init_reductions(&mut self) {
        for depth in 1..64 {
            for move_index in 1..64 {
                let reduction = ((depth as f32).ln() * (move_index as f32).ln() / 2.25).floor() as u32;
                
                let reduction = reduction.max(0).min(depth as u32 - 1);
                
                self.reduction_table[depth][move_index] = reduction;
            }
        }
    }

    pub fn get_reduction(&self, depth: u32, move_index: usize, is_pv: bool) -> u32 {
        if depth < 3 || move_index < 2 {
            return 0;
        }

        let depth_idx = (depth as usize).min(63);
        let move_idx = move_index.min(63);
        let mut reduction = self.reduction_table[depth_idx][move_idx];

        if is_pv {
            reduction = reduction.saturating_sub(1);
        }

        reduction
    }

    pub fn should_reduce(&self, depth: u32, move_index: usize, is_pv: bool, move_score: Score) -> bool {
        if depth < 3 {
            return false;
        }

        if move_index < 2 {
            return false;
        }

        if is_pv && move_index < 4 {
            return false;
        }

        if move_score < -500 {
            return false;
        }

        true
    }
}
