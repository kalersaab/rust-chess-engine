use crate::evaluation::Score;

pub struct FutilityPruning {
    pub margins: Vec<Score>,
}

impl FutilityPruning {
    pub fn new() -> Self {
        FutilityPruning {
            margins: vec![0, 100, 300, 500, 900],
        }
    }

    pub fn can_prune(
        &self,
        depth: u32,
        alpha: Score,
        beta: Score,
        static_eval: Score,
    ) -> bool {
        if depth >= 4 {
            return false;
        }

        if beta - alpha > 1 {
            return false;
        }

        let margin = if (depth as usize) < self.margins.len() {
            self.margins[depth as usize]
        } else {
            self.margins[self.margins.len() - 1]
        };

        static_eval + margin < alpha
    }

    pub fn should_prune_move(
        &self,
        depth: u32,
        alpha: Score,
        static_eval: Score,
        move_value: Score,
    ) -> bool {
        if depth >= 3 {
            return false;
        }

        let threshold = alpha - 100;
        static_eval + move_value < threshold
    }
}

pub struct ReverseFutilityPruning {
    pub margins: Vec<Score>,
}

impl ReverseFutilityPruning {
    pub fn new() -> Self {
        ReverseFutilityPruning {
            margins: vec![0, 200, 500, 1000, 2000],
        }
    }

    pub fn can_prune(
        &self,
        depth: u32,
        beta: Score,
        static_eval: Score,
    ) -> bool {
        if depth >= 5 {
            return false;
        }

        let margin = if (depth as usize) < self.margins.len() {
            self.margins[depth as usize]
        } else {
            self.margins[self.margins.len() - 1]
        };

        static_eval - margin >= beta
    }
}
