import argparse
import json
import sys
from orderbook.serving import Predictor, StreamingPredictor


def main():
    parser = argparse.ArgumentParser(description="Read L2 JSON lines from stdin; emit calibrated signals")
    parser.add_argument("--model", required=True, help="Trusted local .pkl created by project.run")
    args = parser.parse_args()
    stream = StreamingPredictor(Predictor.load(args.model))
    for line in sys.stdin:
        try:
            result = stream.update_safe(json.loads(line))
        except (ValueError, TypeError) as error:
            result = {"status": "error", "direction": 0, "error": str(error)}
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
