from __future__ import annotations

import argparse
import json
from pathlib import Path

from .equivalence import compare_query_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one generated SQL query against a gold query.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--predicted-sql", required=True)
    parser.add_argument("--gold-sql", required=True)
    args = parser.parse_args()
    result = compare_query_results(args.database, args.predicted_sql, args.gold_sql)
    print(json.dumps({"equivalent": result.equivalent, "order_sensitive": result.order_sensitive, "error": result.error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
