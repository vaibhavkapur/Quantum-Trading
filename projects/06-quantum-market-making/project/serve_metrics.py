"""Read-only Prometheus exporter for the latest stored trial per seed/strategy."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from market_making.reporting import collect


def metrics_text(runs):
    groups, failures, _ = collect(runs)
    names = {
        'net_pnl': ('net_pnl','Synthetic currency units'),
        'quote_ack_p99_seconds': ('quote_ack_p99_ns','Local quote-to-ack p99, seconds'),
        'inventory_max': ('max_abs_inventory','Maximum absolute inventory'),
        'fallbacks': ('optimizer_fallbacks','Optimizer fallback count in completed trial'),
        'risk_halted': ('halted','Whether the risk halt was latched'),
    }
    lines=[]
    for name,(field,description) in names.items():
        lines.extend([f'# HELP mm_{name} {description}',f'# TYPE mm_{name} gauge'])
        for key,trials in sorted(groups.items()):
            for trial in trials:
                labels=f'config="{key[0][:12]}",solver="{key[3]}",scenario="{key[4]}",seed="{trial["seed"]}"'
                value=float(trial[field])*(1e-9 if name=='quote_ack_p99_seconds' else 1)
                lines.append(f'mm_{name}{{{labels}}} {value}')
    lines += ['# TYPE mm_failed_trials gauge',f'mm_failed_trials {len(failures)}']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='runs')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=9108)
    args=parser.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/metrics':
                self.send_error(404)
                return
            try:
                data=metrics_text(args.runs).encode()
            except (OSError,ValueError) as exc:
                self.send_error(503,str(exc))
                return
            self.send_response(200)
            self.send_header('Content-Type','text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    print(f'Prometheus endpoint: http://{args.host}:{args.port}/metrics',flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()


if __name__=='__main__':
    main()
