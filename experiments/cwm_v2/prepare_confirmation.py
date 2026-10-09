"""Union exclusions from completed scouts; never copy validation to training."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for path in args.scenes:
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                rows[row["episode_index"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(r) + "\n" for r in rows.values()))
    print(json.dumps({"excluded_episodes": len(rows), "groups": len({r["mirror_group_id"] for r in rows.values()})}))


if __name__ == "__main__":
    main()
