import unittest
from pathlib import Path

import load_path  # noqa: F401

from src.sql_queries import load_named_queries


class SqlQueriesTests(unittest.TestCase):
    def test_load_named_queries_from_analysis_sql(self):
        sql_path = Path(__file__).resolve().parents[1] / "sql" / "analysis.sql"
        queries = load_named_queries(sql_path)

        self.assertEqual(len(queries), 5)
        self.assertTrue(all(stmt.upper().startswith("SELECT") for stmt in queries.values()))
        self.assertIn("1. Merchant Category Volume & Total Spend", queries)


if __name__ == "__main__":
    unittest.main()
