//! Fixed-depth limit order book and marketable-order fills.

pub const MAX_LEVELS: usize = 8;

#[derive(Clone, Copy, Default, Debug)]
pub struct Level {
    pub price: f64,
    pub size: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Book {
    pub bids: [Level; MAX_LEVELS],
    pub asks: [Level; MAX_LEVELS],
}

#[derive(Debug, PartialEq)]
pub struct FillResult {
    pub avg_price: f64,
    /// Quantity that exceeded displayed depth and was priced at the worst level.
    pub beyond_book: f64,
}

impl Book {
    #[inline]
    pub fn update(&mut self, is_ask: bool, level: usize, price: f64, size: f64) {
        let side = if is_ask { &mut self.asks } else { &mut self.bids };
        side[level] = Level { price, size };
    }

    /// Mid price, or None while either side is empty or the book is crossed.
    #[inline]
    pub fn mid(&self) -> Option<f64> {
        let (b, a) = (self.bids[0], self.asks[0]);
        if b.size > 0.0 && a.size > 0.0 && a.price > b.price {
            Some(0.5 * (a.price + b.price))
        } else {
            None
        }
    }

    /// Walk the book with a marketable order. `qty > 0` buys from asks,
    /// `qty < 0` sells into bids. Returns None if the touched side is empty.
    pub fn market_fill(&self, qty: f64) -> Option<FillResult> {
        let side = if qty > 0.0 { &self.asks } else { &self.bids };
        let mut remaining = qty.abs();
        let (mut cost, mut worst) = (0.0, 0.0);
        for lvl in side.iter() {
            if lvl.size <= 0.0 {
                continue;
            }
            let take = remaining.min(lvl.size);
            cost += take * lvl.price;
            worst = lvl.price;
            remaining -= take;
            if remaining <= 0.0 {
                break;
            }
        }
        if worst == 0.0 {
            return None;
        }
        let beyond = remaining.max(0.0);
        cost += beyond * worst;
        Some(FillResult { avg_price: cost / qty.abs(), beyond_book: beyond })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> Book {
        let mut b = Book::default();
        b.update(false, 0, 99.0, 1.0);
        b.update(false, 1, 98.0, 2.0);
        b.update(true, 0, 101.0, 1.0);
        b.update(true, 1, 102.0, 2.0);
        b
    }

    #[test]
    fn mid_requires_both_sides() {
        assert_eq!(Book::default().mid(), None);
        assert_eq!(book().mid(), Some(100.0));
    }

    #[test]
    fn crossed_book_has_no_mid() {
        let mut b = book();
        b.update(true, 0, 98.5, 1.0);
        assert_eq!(b.mid(), None);
    }

    #[test]
    fn buy_walks_asks() {
        let f = book().market_fill(2.0).unwrap();
        assert!((f.avg_price - 101.5).abs() < 1e-12);
        assert_eq!(f.beyond_book, 0.0);
    }

    #[test]
    fn sell_walks_bids() {
        let f = book().market_fill(-3.0).unwrap();
        assert!((f.avg_price - (99.0 + 2.0 * 98.0) / 3.0).abs() < 1e-12);
    }

    #[test]
    fn oversize_order_is_flagged() {
        let f = book().market_fill(5.0).unwrap();
        assert!((f.beyond_book - 2.0).abs() < 1e-12);
        assert!((f.avg_price - (101.0 + 2.0 * 102.0 + 2.0 * 102.0) / 5.0).abs() < 1e-12);
    }

    #[test]
    fn empty_side_cannot_fill() {
        assert!(Book::default().market_fill(1.0).is_none());
    }
}
