import ast
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from src import sql_lab


class SQLLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.database = Path(cls.directory.name) / "test.sqlite"
        with sqlite3.connect(cls.database) as conn:
            conn.execute("""CREATE TABLE upi_transactions (
                amount REAL, timestamp TEXT, merchant_category TEXT,
                location TEXT, upi_channel TEXT, risk_flag INTEGER, sender_id TEXT
            )""")
            conn.executemany(
                "INSERT INTO upi_transactions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(10.5, "2026-01-01 12:00:00", "Food", "Delhi", "App", 0, "USER1")] * 1100,
            )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_query(self, sql):
        return sql_lab.execute_query(self.database, sql)

    def test_presets_still_work(self):
        dashboard = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
        tree = ast.parse(dashboard.read_text())
        presets = next(
            ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "queries" for target in node.targets)
        )
        for name, query in presets.items():
            with self.subTest(preset=name):
                self.assertTrue(self.run_query(query)["rows"])

    def test_values_empty_result_and_row_boundary(self):
        result = self.run_query("SELECT NULL AS missing, 1 AS number, '₹' AS text, x'00ff' AS bytes")
        self.assertEqual(result, {"columns": ["missing", "number", "text", "bytes"],
                                  "rows": [[None, 1, "₹", b"\x00\xff"]]})
        self.assertEqual(self.run_query("SELECT amount FROM upi_transactions WHERE 0"),
                         {"columns": ["amount"], "rows": []})
        self.assertEqual(len(self.run_query("SELECT * FROM upi_transactions LIMIT 1000")["rows"]), 1000)

    def test_row_and_total_byte_limits(self):
        for query, message in [
            ("SELECT * FROM upi_transactions", "row limit"),
            ("SELECT hex(zeroblob(1000)) FROM upi_transactions LIMIT 600", "result limit"),
            ("SELECT zeroblob(1048576)", "result limit"),
        ]:
            with self.subTest(query=query), self.assertRaisesRegex(sql_lab.SQLLabError, message):
                self.run_query(query)

    def test_expensive_single_row_aggregate_is_interrupted(self):
        start = time.monotonic()
        with self.assertRaises(sql_lab.SQLLabError):
            self.run_query("SELECT SUM(a.amount + b.amount + c.amount) FROM upi_transactions a, "
                           "upi_transactions b, upi_transactions c")
        self.assertLess(time.monotonic() - start, sql_lab.WORKER_SECONDS + 1)
        self.assertEqual(self.run_query("SELECT 1")["rows"], [[1]])

    def test_large_single_allocation_is_confined_and_next_query_works(self):
        with self.assertRaises(sql_lab.SQLLabError):
            self.run_query("SELECT randomblob(1000000000)")
        self.assertEqual(self.run_query("SELECT COUNT(*) FROM upi_transactions")["rows"], [[1100]])

    def test_controls_cannot_be_changed_and_snapshot_cannot_be_mutated(self):
        for query in [
            "PRAGMA query_only = OFF", "PRAGMA hard_heap_limit = 1",
            "ATTACH DATABASE ':memory:' AS other", "DELETE FROM upi_transactions",
            "CREATE TABLE other (x)", "SELECT load_extension('missing')",
            "SELECT 1; SELECT 2;",
        ]:
            with self.subTest(query=query), self.assertRaises(sql_lab.SQLLabError):
                self.run_query(query)
        self.assertEqual(self.run_query("SELECT COUNT(*) FROM upi_transactions")["rows"], [[1100]])

    def test_sql_size_and_busy_limits(self):
        with self.assertRaisesRegex(sql_lab.SQLLabError, "query limit"):
            self.run_query("SELECT '" + "₹" * 30000 + "'")
        with sql_lab._WORKER_SLOTS, sql_lab._WORKER_SLOTS:
            with self.assertRaisesRegex(sql_lab.SQLLabError, "busy"):
                self.run_query("SELECT 1")
        self.assertEqual(self.run_query("SELECT 1")["rows"], [[1]])

    def test_parent_timeout_kills_worker_and_releases_slot(self):
        children = []
        original_popen = subprocess.Popen

        def capture_child(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            return child

        with patch.object(sql_lab, "WORKER_SECONDS", 0.001), \
                patch.object(subprocess, "Popen", side_effect=capture_child):
            with self.assertRaisesRegex(sql_lab.SQLLabError, "time limit"):
                self.run_query("WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x")
        self.assertIsNotNone(children[0].poll())
        self.assertEqual(self.run_query("SELECT 1")["rows"], [[1]])


if __name__ == "__main__":
    unittest.main()
