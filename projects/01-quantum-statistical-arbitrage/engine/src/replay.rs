//! Event replay, signal state machine, simulated order latency and P&L.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use crate::book::{Book, MAX_LEVELS};

#[derive(Clone, Debug)]
pub struct Params {
    pub quotes: String,
    pub weights: Vec<f64>,
    pub warmup_until_ns: i64,
    pub entry: f64,
    pub exit: f64,
    pub stop: f64,
    pub notional: f64,
    pub latency_ns: i64,
    pub fee_bps: f64,
    pub cooldown_ticks: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            quotes: String::new(),
            weights: Vec::new(),
            warmup_until_ns: 0,
            entry: 2.0,
            exit: 0.5,
            stop: 4.0,
            notional: 100_000.0,
            latency_ns: 500_000,
            fee_bps: 1.0,
            cooldown_ticks: 200,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    pub ts_ns: i64,
    pub sym: usize,
    pub is_ask: bool,
    pub level: usize,
    pub price: f64,
    pub size: f64,
}

/// Parse `ts_ns,sym,side,level,price,size` (side: 0 = bid, 1 = ask).
pub fn parse_line(line: &str) -> Result<Event, String> {
    let mut it = line.trim().split(',');
    let mut next = |name: &str| it.next().ok_or_else(|| format!("missing field {name}"));
    let ts_ns = next("ts_ns")?.parse::<i64>().map_err(|e| format!("ts_ns: {e}"))?;
    let sym = next("sym")?.parse::<usize>().map_err(|e| format!("sym: {e}"))?;
    let side = next("side")?.parse::<u8>().map_err(|e| format!("side: {e}"))?;
    let level = next("level")?.parse::<usize>().map_err(|e| format!("level: {e}"))?;
    let price = next("price")?.parse::<f64>().map_err(|e| format!("price: {e}"))?;
    let size = next("size")?.parse::<f64>().map_err(|e| format!("size: {e}"))?;
    if side > 1 {
        return Err(format!("side must be 0 or 1, got {side}"));
    }
    if level >= MAX_LEVELS {
        return Err(format!("level {level} exceeds supported depth {MAX_LEVELS}"));
    }
    if !(price > 0.0) || !(size >= 0.0) {
        return Err("price must be positive and size non-negative".to_string());
    }
    Ok(Event { ts_ns, sym, is_ask: side == 1, level, price, size })
}

pub fn load(path: &str) -> Result<Vec<Event>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let mut events = Vec::new();
    let mut last_ts = i64::MIN;
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read error at line {}: {e}", n + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let ev = parse_line(&line).map_err(|e| format!("line {}: {e}", n + 1))?;
        if ev.ts_ns < last_ts {
            return Err(format!("line {}: timestamps must be non-decreasing", n + 1));
        }
        last_ts = ev.ts_ns;
        events.push(ev);
    }
    if events.is_empty() {
        return Err("no events".to_string());
    }
    Ok(events)
}

/// Z-score state machine: returns the desired spread direction (-1, 0, +1).
#[derive(Debug, Default)]
pub struct Signal {
    pub dir: i8,
    pub cooldown: u64,
    pub entries: u64,
    pub stops: u64,
}

impl Signal {
    #[inline]
    pub fn step(&mut self, z: f64, p: &Params) -> i8 {
        if self.dir == 0 {
            if self.cooldown > 0 {
                self.cooldown -= 1;
            } else if z > p.entry && z < p.stop {
                self.dir = -1;
                self.entries += 1;
            } else if z < -p.entry && z > -p.stop {
                self.dir = 1;
                self.entries += 1;
            }
        } else if z.abs() >= p.stop {
            self.dir = 0;
            self.stops += 1;
            self.cooldown = p.cooldown_ticks;
        } else if (self.dir == 1 && z >= -p.exit) || (self.dir == -1 && z <= p.exit) {
            self.dir = 0;
        }
        self.dir
    }
}

struct Pending {
    due_ns: i64,
    target: Vec<f64>,
}

#[derive(Debug, Default)]
pub struct Report {
    pub events: usize,
    pub ticks: u64,
    pub warmup_ticks: u64,
    pub traded_ticks: u64,
    pub load_seconds: f64,
    pub replay_seconds: f64,
    pub events_per_second: f64,
    pub spread_mean: f64,
    pub spread_std: f64,
    pub entries: u64,
    pub stops: u64,
    pub orders: u64,
    pub fills: u64,
    pub unfilled: u64,
    pub beyond_book_qty: f64,
    pub traded_notional: f64,
    pub fees: f64,
    pub slippage_vs_mid: f64,
    pub pnl: f64,
    pub notional: f64,
    pub latency_ns: i64,
    pub tick_to_decision_ns: [u64; 5], // p50, p90, p99, p99.9, max
    pub tick_to_order_ns: [u64; 5],
    pub tick_to_order_samples: usize,
}

fn percentiles(samples: &mut [u64]) -> [u64; 5] {
    if samples.is_empty() {
        return [0; 5];
    }
    samples.sort_unstable();
    let at = |q: f64| samples[(((samples.len() - 1) as f64) * q).round() as usize];
    [at(0.50), at(0.90), at(0.99), at(0.999), samples[samples.len() - 1]]
}

fn arr(a: &[u64; 5]) -> String {
    format!("{{\"p50\": {}, \"p90\": {}, \"p99\": {}, \"p99_9\": {}, \"max\": {}}}", a[0], a[1], a[2], a[3], a[4])
}

impl Report {
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"events\": {},\n  \"ticks\": {},\n  \"warmup_ticks\": {},\n  \"traded_ticks\": {},\n  \"load_seconds\": {:.6},\n  \"replay_seconds\": {:.6},\n  \"events_per_second\": {:.1},\n  \"spread_mean\": {:.10},\n  \"spread_std\": {:.10},\n  \"entries\": {},\n  \"stops\": {},\n  \"orders\": {},\n  \"fills\": {},\n  \"unfilled\": {},\n  \"beyond_book_qty\": {:.6},\n  \"traded_notional\": {:.2},\n  \"fees\": {:.4},\n  \"slippage_vs_mid\": {:.4},\n  \"pnl\": {:.4},\n  \"notional\": {:.2},\n  \"simulated_order_latency_ns\": {},\n  \"tick_to_decision_ns\": {},\n  \"tick_to_order_ns\": {},\n  \"tick_to_order_samples\": {}\n}}\n",
            self.events, self.ticks, self.warmup_ticks, self.traded_ticks, self.load_seconds, self.replay_seconds,
            self.events_per_second, self.spread_mean, self.spread_std, self.entries, self.stops, self.orders,
            self.fills, self.unfilled, self.beyond_book_qty, self.traded_notional, self.fees, self.slippage_vs_mid,
            self.pnl, self.notional, self.latency_ns, arr(&self.tick_to_decision_ns), arr(&self.tick_to_order_ns),
            self.tick_to_order_samples
        )
    }
}

struct Portfolio {
    cash: f64,
    pos: Vec<f64>,
}

/// Fill `target - pos` against the books; returns the number of legs that could not fill.
fn execute(books: &[Book], target: &[f64], pf: &mut Portfolio, fee_bps: f64, rep: &mut Report) -> u64 {
    let mut unfilled = 0;
    for (i, book) in books.iter().enumerate() {
        let qty = target[i] - pf.pos[i];
        if qty.abs() < 1e-12 {
            continue;
        }
        match (book.market_fill(qty), book.mid()) {
            (Some(fill), Some(mid)) => {
                let notional = qty.abs() * fill.avg_price;
                let fee = notional * fee_bps / 1e4;
                pf.cash -= qty * fill.avg_price + fee;
                pf.pos[i] += qty;
                rep.fills += 1;
                rep.fees += fee;
                rep.traded_notional += notional;
                rep.slippage_vs_mid += qty * (fill.avg_price - mid);
                rep.beyond_book_qty += fill.beyond_book;
            }
            _ => unfilled += 1,
        }
    }
    unfilled
}

pub fn run(p: &Params) -> Result<Report, String> {
    let t_load = Instant::now();
    let events = load(&p.quotes)?;
    let load_seconds = t_load.elapsed().as_secs_f64();
    let mut rep = replay(&events, p)?;
    rep.load_seconds = load_seconds;
    Ok(rep)
}

pub fn replay(events: &[Event], p: &Params) -> Result<Report, String> {
    let n = p.weights.len();
    if let Some(bad) = events.iter().find(|e| e.sym >= n) {
        return Err(format!("event references symbol {} but only {} weights were given", bad.sym, n));
    }
    let mut books = vec![Book::default(); n];
    let mut mids = vec![0.0f64; n];
    let mut have = vec![false; n];
    let mut pf = Portfolio { cash: 0.0, pos: vec![0.0; n] };
    let mut signal = Signal::default();
    let mut pending: Vec<Pending> = Vec::new();
    let mut rep = Report { events: events.len(), notional: p.notional, latency_ns: p.latency_ns, ..Default::default() };

    // Welford accumulators for the warm-up (train-only) spread statistics.
    let (mut count, mut mean, mut m2) = (0u64, 0.0f64, 0.0f64);
    let mut std = 0.0f64;
    let mut frozen = false;
    let mut decision_ns: Vec<u64> = Vec::with_capacity(events.len() / 8);
    let mut order_ns: Vec<u64> = Vec::new();

    let t_replay = Instant::now();
    let mut i = 0;
    while i < events.len() {
        let ts = events[i].ts_ns;
        // Orders whose simulated latency has elapsed see the book as it was before this tick.
        if !pending.is_empty() && pending[0].due_ns <= ts {
            let due: Vec<Pending> = pending.drain(..).collect();
            for order in due {
                if order.due_ns <= ts {
                    rep.unfilled += execute(&books, &order.target, &mut pf, p.fee_bps, &mut rep);
                } else {
                    pending.push(order);
                }
            }
        }

        let t0 = Instant::now();
        // Apply every update carrying this timestamp, then evaluate the tick once.
        while i < events.len() && events[i].ts_ns == ts {
            let e = &events[i];
            books[e.sym].update(e.is_ask, e.level, e.price, e.size);
            if let Some(m) = books[e.sym].mid() {
                mids[e.sym] = m;
                have[e.sym] = true;
            }
            i += 1;
        }
        rep.ticks += 1;
        if !have.iter().all(|&h| h) {
            continue;
        }
        let spread: f64 = p.weights.iter().zip(&mids).map(|(w, m)| w * m.ln()).sum();

        if ts <= p.warmup_until_ns {
            count += 1;
            let d = spread - mean;
            mean += d / count as f64;
            m2 += d * (spread - mean);
            rep.warmup_ticks += 1;
            continue;
        }
        if !frozen {
            if count < 30 {
                return Err("warm-up segment has fewer than 30 complete ticks".to_string());
            }
            std = (m2 / (count - 1) as f64).sqrt();
            if !(std > 0.0) {
                return Err("spread has zero variance on the warm-up segment".to_string());
            }
            frozen = true;
        }
        rep.traded_ticks += 1;
        let prev = signal.dir;
        let dir = signal.step((spread - mean) / std, p);
        let mut sent = false;
        if dir != prev {
            let target: Vec<f64> =
                p.weights.iter().zip(&mids).map(|(w, m)| dir as f64 * w * p.notional / m).collect();
            pending.push(Pending { due_ns: ts + p.latency_ns, target });
            rep.orders += 1;
            sent = true;
        }
        let elapsed = t0.elapsed().as_nanos() as u64;
        decision_ns.push(elapsed);
        if sent {
            order_ns.push(elapsed);
        }
    }
    rep.replay_seconds = t_replay.elapsed().as_secs_f64();

    // Liquidate at the final book so P&L is fully realised and pays costs.
    let flat = vec![0.0; n];
    rep.unfilled += execute(&books, &flat, &mut pf, p.fee_bps, &mut rep);
    let residual: f64 = pf.pos.iter().zip(&mids).map(|(q, m)| q * m).sum();
    rep.pnl = pf.cash + residual;
    rep.entries = signal.entries;
    rep.stops = signal.stops;
    rep.spread_mean = mean;
    rep.spread_std = std;
    rep.events_per_second = if rep.replay_seconds > 0.0 { events.len() as f64 / rep.replay_seconds } else { 0.0 };
    rep.tick_to_order_samples = order_ns.len();
    rep.tick_to_decision_ns = percentiles(&mut decision_ns);
    rep.tick_to_order_ns = percentiles(&mut order_ns);
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params { weights: vec![0.5, -0.5], latency_ns: 1, fee_bps: 0.0, cooldown_ticks: 0, ..Default::default() }
    }

    /// One tick: both sides of `sym` around `mid` with a tight spread and deep size.
    fn tick(ts: i64, sym: usize, mid: f64) -> Vec<Event> {
        vec![
            Event { ts_ns: ts, sym, is_ask: false, level: 0, price: mid - 0.005, size: 1e9 },
            Event { ts_ns: ts, sym, is_ask: true, level: 0, price: mid + 0.005, size: 1e9 },
        ]
    }

    #[test]
    fn parses_a_valid_line() {
        let e = parse_line("1700000000000000001,2,1,3,101.25,0.5").unwrap();
        assert_eq!(e, Event { ts_ns: 1700000000000000001, sym: 2, is_ask: true, level: 3, price: 101.25, size: 0.5 });
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(parse_line("1,0,2,0,10,1").is_err()); // bad side
        assert!(parse_line("1,0,0,99,10,1").is_err()); // level too deep
        assert!(parse_line("1,0,0,0,-5,1").is_err()); // negative price
        assert!(parse_line("1,0,0").is_err()); // too few fields
    }

    #[test]
    fn signal_enters_exits_and_stops() {
        let p = params();
        let mut s = Signal::default();
        assert_eq!(s.step(1.0, &p), 0);
        assert_eq!(s.step(2.5, &p), -1);
        assert_eq!(s.step(1.0, &p), -1);
        assert_eq!(s.step(0.4, &p), 0);
        assert_eq!(s.step(-2.5, &p), 1);
        assert_eq!(s.step(-4.5, &p), 0);
        assert_eq!((s.entries, s.stops), (2, 1));
        // Beyond the stop band no new position is opened.
        assert_eq!(s.step(5.0, &p), 0);
    }

    #[test]
    fn mean_reverting_spread_makes_money_and_uses_only_warmup_stats() {
        // Symbol 1 is constant; symbol 0 oscillates during warm-up, then spikes and reverts.
        let mut ev = Vec::new();
        let mut ts = 0i64;
        ev.extend(tick(ts, 1, 100.0));
        for k in 0..200 {
            ts += 10;
            let m = 100.0 * (1.0 + 0.001 * if k % 2 == 0 { 1.0 } else { -1.0 });
            ev.extend(tick(ts, 0, m));
        }
        let warmup_until = ts;
        for m in [100.3, 100.3, 100.0, 100.0] {
            ts += 10;
            ev.extend(tick(ts, 0, m));
        }
        let mut p = params();
        p.warmup_until_ns = warmup_until;
        let rep = replay(&ev, &p).unwrap();
        assert_eq!(rep.warmup_ticks, 200);
        assert_eq!(rep.entries, 1);
        assert_eq!(rep.orders, 2); // entry and exit
        assert!(rep.spread_std > 0.0004 && rep.spread_std < 0.0006);
        // Short 0.5 * notional of symbol 0 at ~100.3, cover at ~100.0, minus crossing the spread.
        assert!(rep.pnl > 100.0, "pnl = {}", rep.pnl);
        assert_eq!(rep.unfilled, 0);
    }

    #[test]
    fn orders_respect_simulated_latency() {
        // With a latency longer than the remaining stream, the entry never fills before liquidation.
        let mut ev = Vec::new();
        let mut ts = 0i64;
        ev.extend(tick(ts, 1, 100.0));
        for k in 0..100 {
            ts += 10;
            ev.extend(tick(ts, 0, if k % 2 == 0 { 100.1 } else { 99.9 }));
        }
        let warmup_until = ts;
        ts += 10;
        ev.extend(tick(ts, 0, 100.3));
        let mut p = params();
        p.warmup_until_ns = warmup_until;
        p.latency_ns = 1_000_000;
        let rep = replay(&ev, &p).unwrap();
        assert_eq!(rep.orders, 1);
        assert_eq!(rep.fills, 0);
        assert_eq!(rep.pnl, 0.0);
    }

    #[test]
    fn unknown_symbol_is_an_error() {
        let ev = tick(0, 5, 100.0);
        assert!(replay(&ev, &params()).is_err());
    }
}
