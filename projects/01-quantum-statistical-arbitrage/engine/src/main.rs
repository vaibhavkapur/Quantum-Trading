//! L2 quote replay + spread signal + execution-cost simulator.
//!
//! This is the latency-critical, purely classical side of the project. It
//! replays level-by-level book updates, maintains per-symbol books, computes a
//! weighted log-mid spread, runs a z-score state machine and sends simulated
//! market orders that fill against the book as it stands `latency_ns` after the
//! decision. Hedge weights come from the research layer (Python); mean and
//! standard deviation of the spread are estimated on the warm-up segment only.
//!
//! Reported latency is the wall-clock time this process needs from "a tick's
//! last update is applied" to "an order decision exists". It excludes network,
//! kernel and exchange time and is not a claim about any live system.

mod book;
mod replay;

use std::env;
use std::fs;
use std::process::ExitCode;

use replay::{run, Params};

fn usage() -> &'static str {
    "usage: statarb-engine replay --quotes FILE --weights W0,W1,... --warmup-until-ns TS\n\
     \x20      [--entry 2.0] [--exit 0.5] [--stop 4.0] [--notional 100000]\n\
     \x20      [--latency-ns 500000] [--fee-bps 1.0] [--cooldown-ticks 200] [--out FILE]"
}

fn parse_args(args: &[String]) -> Result<(Params, Option<String>), String> {
    if args.first().map(String::as_str) != Some("replay") {
        return Err(usage().to_string());
    }
    let mut p = Params::default();
    let mut out = None;
    let mut have = (false, false, false);
    let mut i = 1;
    while i < args.len() {
        let key = args[i].as_str();
        let val = args.get(i + 1).ok_or_else(|| format!("missing value for {key}"))?;
        let num = |v: &str| v.parse::<f64>().map_err(|_| format!("bad number for {key}: {v}"));
        match key {
            "--quotes" => {
                p.quotes = val.clone();
                have.0 = true;
            }
            "--weights" => {
                p.weights = val.split(',').map(num).collect::<Result<Vec<_>, _>>()?;
                have.1 = true;
            }
            "--warmup-until-ns" => {
                p.warmup_until_ns = val.parse().map_err(|_| format!("bad timestamp: {val}"))?;
                have.2 = true;
            }
            "--entry" => p.entry = num(val)?,
            "--exit" => p.exit = num(val)?,
            "--stop" => p.stop = num(val)?,
            "--notional" => p.notional = num(val)?,
            "--fee-bps" => p.fee_bps = num(val)?,
            "--latency-ns" => p.latency_ns = val.parse().map_err(|_| format!("bad latency: {val}"))?,
            "--cooldown-ticks" => p.cooldown_ticks = val.parse().map_err(|_| format!("bad cooldown: {val}"))?,
            "--out" => out = Some(val.clone()),
            _ => return Err(format!("unknown argument {key}\n{}", usage())),
        }
        i += 2;
    }
    if !(have.0 && have.1 && have.2) {
        return Err(usage().to_string());
    }
    if !(p.exit < p.entry && p.entry < p.stop) {
        return Err("thresholds must satisfy exit < entry < stop".to_string());
    }
    if p.weights.len() < 2 {
        return Err("need at least two weights".to_string());
    }
    Ok((p, out))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let (params, out) = match parse_args(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&params) {
        Ok(report) => {
            let json = report.to_json();
            match out {
                Some(path) => {
                    if let Err(e) = fs::write(&path, &json) {
                        eprintln!("cannot write {path}: {e}");
                        return ExitCode::from(1);
                    }
                }
                None => println!("{json}"),
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
