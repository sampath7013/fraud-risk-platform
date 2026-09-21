import sqlite3


DATABASE = "fraud_risk.db"


def print_rows(title, rows):
    """
    Print database rows in a readable dictionary format.
    """

    print(f"\n{title}")
    print("-" * 70)

    if not rows:
        print("No records found.")
        return

    for row in rows:
        print(dict(row))


def main():
    # -----------------------------------------------------
    # Database Connection
    # -----------------------------------------------------

    connection = sqlite3.connect(DATABASE)

    # Allows columns to be accessed by name.
    #
    # Without this:
    # row[0], row[1], row[2]
    #
    # With sqlite3.Row:
    # row["transaction_id"]
    # row["amount"]
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    # =====================================================
    # CREATE CUSTOMERS TABLE
    # =====================================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            segment TEXT NOT NULL
        );
        """
    )

    # -----------------------------------------------------
    # Insert Sample Customers
    # -----------------------------------------------------

    customers = [
        ("CUST-100", "Alice", "PREMIUM"),
        ("CUST-200", "Bob", "STANDARD"),
        ("CUST-300", "Charlie", "PREMIUM"),
        ("CUST-400", "David", "STANDARD"),
    ]

    cursor.executemany(
        """
        INSERT OR IGNORE INTO customers (
            customer_id,
            name,
            segment
        )
        VALUES (?, ?, ?);
        """,
        customers,
    )

    connection.commit()

    # =====================================================
    # 1. SELECT ALL
    # =====================================================

    cursor.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY id DESC;
        """
    )

    rows = cursor.fetchall()

    print_rows(
        "1. All Transactions",
        rows,
    )

    # =====================================================
    # 2. WHERE
    # =====================================================

    cursor.execute(
        """
        SELECT *
        FROM transactions
        WHERE decision = ?;
        """,
        ("BLOCK",),
    )

    rows = cursor.fetchall()

    print_rows(
        "2. BLOCK Transactions",
        rows,
    )

    # =====================================================
    # 3. MULTIPLE WHERE CONDITIONS
    # =====================================================

    cursor.execute(
        """
        SELECT *
        FROM transactions
        WHERE decision = ?
          AND amount >= ?;
        """,
        ("BLOCK", 5000),
    )

    rows = cursor.fetchall()

    print_rows(
        "3. BLOCK Transactions >= $5000",
        rows,
    )

    # =====================================================
    # 4. AGGREGATION + GROUP BY
    # =====================================================

    cursor.execute(
        """
        SELECT
            decision,
            COUNT(*) AS transaction_count,
            AVG(amount) AS average_amount,
            MAX(amount) AS maximum_amount,
            MIN(amount) AS minimum_amount
        FROM transactions
        GROUP BY decision;
        """
    )

    rows = cursor.fetchall()

    print_rows(
        "4. Statistics By Decision",
        rows,
    )

    # =====================================================
    # 5. LIMIT + OFFSET
    # =====================================================

    cursor.execute(
        """
        SELECT *
        FROM transactions
        ORDER BY id DESC
        LIMIT ?
        OFFSET ?;
        """,
        (10, 0),
    )

    rows = cursor.fetchall()

    print_rows(
        "5. Paginated Transactions",
        rows,
    )

    # =====================================================
    # 6. GROUP BY + HAVING
    # =====================================================

    cursor.execute(
        """
        SELECT
            decision,
            COUNT(*) AS transaction_count,
            AVG(amount) AS average_amount
        FROM transactions
        GROUP BY decision
        HAVING COUNT(*) >= ?;
        """,
        (2,),
    )

    rows = cursor.fetchall()

    print_rows(
        "6. Decision Groups With At Least 2 Transactions",
        rows,
    )

    # =====================================================
    # 7. INNER JOIN
    # =====================================================

    cursor.execute(
        """
        SELECT
            t.transaction_id,
            t.amount,
            t.decision,
            c.customer_id,
            c.name,
            c.segment
        FROM transactions AS t
        INNER JOIN customers AS c
            ON t.customer_id = c.customer_id
        ORDER BY t.id;
        """
    )

    rows = cursor.fetchall()

    print_rows(
        "7. Transactions With Customer Information",
        rows,
    )

    # =====================================================
    # 8. LEFT JOIN
    # =====================================================

    cursor.execute(
        """
        SELECT
            t.transaction_id,
            t.customer_id,
            t.amount,
            t.decision,
            c.name,
            c.segment
        FROM transactions AS t
        LEFT JOIN customers AS c
            ON t.customer_id = c.customer_id
        ORDER BY t.id;
        """
    )

    rows = cursor.fetchall()

    print_rows(
        "8. All Transactions With Optional Customer Information",
        rows,
    )

    # =====================================================
    # 9. CUSTOMER TRANSACTION SUMMARY
    # =====================================================

    cursor.execute(
        """
        SELECT
            c.customer_id,
            c.name,
            c.segment,
            COUNT(t.id) AS transaction_count,
            COALESCE(SUM(t.amount), 0) AS total_amount,
            COALESCE(AVG(t.amount), 0) AS average_amount
        FROM customers AS c
        LEFT JOIN transactions AS t
            ON c.customer_id = t.customer_id
        GROUP BY
            c.customer_id,
            c.name,
            c.segment
        ORDER BY total_amount DESC;
        """
    )

    rows = cursor.fetchall()

    print_rows(
        "9. Customer Transaction Summary",
        rows,
    )

    # =====================================================
    # Close Database Connection
    # =====================================================

    connection.close()


if __name__ == "__main__":
    main()