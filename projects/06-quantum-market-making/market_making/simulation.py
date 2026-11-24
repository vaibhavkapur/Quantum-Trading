from __future__ import annotations

from dataclasses import asdict
import math
import time
import numpy as np
from .native import Book
from .policy import adaptive_policy


def simulate(frames, config, policy, controller=None, history=()):
    if not frames:
        raise ValueError("cannot simulate empty replay")
    e = config["execution"]
    tick = e["tick_value"]
    equity, inventories, rows, fills, native_ns, ack_ns, policies = [], [], [], [], [], [], []
    moves = [history[j]["mid_ticks"] - history[j-1]["mid_ticks"] for j in range(1, len(history))][-50:]
    previous_mid = frames[0]["mid_ticks"]
    quoted_frames = filled_frames = 0
    start = time.perf_counter()
    with Book(e) as book:
        for i, frame in enumerate(frames):
            if controller is not None and i and i % config["optimizer"]["update_interval"] == 0:
                past = [*history, *frames[:i]][-config["optimizer"]["rolling_window"]:]
                policy = controller.update(past, policy, frame["timestamp_ns"])
            mid, ts = frame["mid_ticks"], frame["timestamp_ns"]
            before = book.stats
            # Flow belongs to the interval ending at this frame. Execute against the prior book.
            if i:
                sides = (1, -1) if frame["buy_first"] else (-1, 1)
                for side in sides:
                    qty = frame["market_buy_qty" if side == 1 else "market_sell_qty"]
                    if qty:
                        book.add(10**12 + i*2 + (side==1), side, 1, qty, timestamp_ns=ts, market=True)
            interval_fills = book.fills_since(before.fills)
            if i and (before.open_buy or before.open_sell):
                quoted_frames += 1
                filled_frames += bool(interval_fills)
            for fill in interval_fills:
                fill.update(frame_index=i, reference_mid_ticks=previous_mid)
                fills.append(fill)
            # Market flow precedes the observed mark; moving quotes are cancelled after that flow.
            if mid != previous_mid:
                book.cancel_all()
            book.reconcile(frame)
            moves.append(mid - previous_mid)
            sigma = float(np.std(moves[-50:])) if len(moves) > 1 else 0.0
            active = adaptive_policy(policy, sigma, 1 - i / len(frames))
            begin = time.perf_counter_ns()
            native_ns.append(book.quote(ts, mid, active, sigma))
            ack_ns.append(time.perf_counter_ns() - begin)
            stats = book.stats
            value = stats.cash + stats.inventory * mid * tick
            equity.append(value)
            inventories.append(stats.inventory)
            rows.append(dict(timestamp_ns=ts, mid_ticks=mid, equity=value, inventory=stats.inventory,
                             sigma_ticks=sigma, policy_version=policy.version, halted=stats.halted,
                             native_quote_ns=native_ns[-1], quote_ack_ns=ack_ns[-1]))
            if not policies or policies[-1]["version"] != policy.version:
                policies.append(dict(timestamp_ns=ts, **asdict(policy)))
            previous_mid = mid
        before = book.stats.fills
        book.flatten(frames[-1]["timestamp_ns"])
        for fill in book.fills_since(before):
            fill.update(frame_index=len(frames)-1, reference_mid_ticks=previous_mid)
            fills.append(fill)
        final = book.stats
        final_equity = final.cash + final.inventory * previous_mid * tick
        equity.append(final_equity)
        rows.append(dict(timestamp_ns=frames[-1]["timestamp_ns"], mid_ticks=previous_mid, equity=final_equity,
                         inventory=final.inventory, sigma_ticks=rows[-1]["sigma_ticks"], policy_version=policy.version,
                         halted=final.halted, native_quote_ns=0, quote_ack_ns=0))
    gross_spread, adverse, censored, maker_qty, taker_qty = 0.0, 0.0, 0, 0, 0
    for fill in fills:
        signed_qty = fill["side"] * fill["quantity"]
        fill["spread_capture"] = signed_qty * (fill["reference_mid_ticks"] - fill["price"]) * tick
        horizon = fill["frame_index"] + e["adverse_horizon"]
        fill["adverse_selection_cost"] = None
        if fill["maker"]:
            maker_qty += fill["quantity"]
            gross_spread += fill["spread_capture"]
            if horizon < len(frames):
                cost = -signed_qty * (frames[horizon]["mid_ticks"] - fill["reference_mid_ticks"]) * tick
                fill["adverse_selection_cost"] = cost
                adverse += cost
            else:
                censored += 1
        else:
            taker_qty += fill["quantity"]
    curve = np.asarray([0.0, *equity])
    changes = np.diff(curve)
    std = float(np.std(changes, ddof=1)) if len(changes)>1 else 0.0
    inv = np.asarray(inventories)
    metrics = dict(
        net_pnl=final_equity, gross_spread_capture=gross_spread,
        net_spread_capture=gross_spread-final.maker_fees-final.cancels*e["cancel_cost"],
        maker_fees=final.maker_fees, maker_rebates=max(0.0, -final.maker_fees), taker_fees=final.taker_fees,
        cancellation_cost=final.cancels*e["cancel_cost"], adverse_selection_cost=adverse,
        censored_markouts=censored, maker_quantity=maker_qty, taker_quantity=taker_qty, fill_count=len(fills),
        fill_probability=filled_frames/quoted_frames if quoted_frames else 0.0, quoted_intervals=quoted_frames,
        sharpe=math.sqrt(len(changes))*float(np.mean(changes))/std if std > 1e-12 else 0.0,
        max_drawdown=float(np.max(np.maximum.accumulate(curve)-curve)),
        mean_abs_inventory=float(np.mean(np.abs(inv))), mean_squared_inventory=float(np.mean(inv**2)),
        max_abs_inventory=int(np.max(np.abs(inv))), inventory_p05=float(np.quantile(inv,.05)),
        inventory_p50=float(np.quantile(inv,.5)), inventory_p95=float(np.quantile(inv,.95)),
        residual_inventory=final.inventory, halted=bool(final.halted), cancels=final.cancels,
        native_quote_p99_ns=float(np.quantile(native_ns,.99)), quote_ack_p99_ns=float(np.quantile(ack_ns,.99)),
        simulation_seconds=time.perf_counter()-start,
    )
    return dict(metrics=metrics, trajectory=rows, fills=fills, policies=policies)


def policy_cost(result, config, steps):
    """Minimize negative per-frame net P&L plus inventory and toxic-fill penalties."""
    m, o = result["metrics"], config["optimizer"]
    return (-m["net_pnl"] + o["adverse_penalty"]*max(0.0, m["adverse_selection_cost"])) / steps + o["inventory_penalty"]*m["mean_squared_inventory"]
