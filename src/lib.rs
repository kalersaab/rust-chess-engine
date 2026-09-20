pub mod board;
pub mod movegen;
pub mod evaluation;
pub mod search;
pub mod move_ordering;
pub mod transposition_table;
pub mod uci;
pub mod perft;
pub mod opening_book;

pub use board::Board;
pub use evaluation::Evaluator;
pub use search::Searcher;
pub use uci::UciEngine;

pub mod prelude {
    pub use crate::board::{Board, Color};
    pub use crate::evaluation::Evaluator;
    pub use crate::search::Searcher;
    pub use crate::uci::UciEngine;
}
