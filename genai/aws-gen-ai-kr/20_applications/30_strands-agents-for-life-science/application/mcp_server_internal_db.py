# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import logging
import sys
from mcp.server.fastmcp import FastMCP
import psycopg2
from collections import defaultdict
import os

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "YOUR_RDS_ENDPOINT"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "database": os.environ.get("DB_NAME", "agentdb"),
    "user": os.environ.get("DB_USER", "dbadmin"),
    "password": os.environ.get("DB_PASSWORD", "postgres")
}

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("internal_db_mcp")

try:
    mcp = FastMCP(name="internal_db_tool")
    logger.info("internal database Postgres MCP server initialized successfully")
except Exception as e:
    logger.error(f"Error: {str(e)}")

@mcp.tool()
async def fetch_table_schema() -> dict:
    """Postgres public 스키마의 테이블과 컬럼 정보를 조회하는 도구입니다.
    - 쿼리를 자동 생성할 때 테이블 구조가 필요하면 이 도구를 사용하세요.
    - 단순히 데이터 조회만 하는 경우에는 호출할 필요가 없습니다.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                           SELECT
                               c.relname AS table_name,
                               a.attname AS column_name,
                               pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
                               d.description AS column_comment
                           FROM pg_catalog.pg_attribute a
                                    JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
                                    JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                                    LEFT JOIN pg_catalog.pg_description d ON a.attrelid = d.objoid AND a.attnum = d.objsubid
                           WHERE a.attnum > 0 AND NOT a.attisdropped
                             AND n.nspname = 'public'
                             AND c.relkind = 'r'
                           ORDER BY c.relname, a.attnum;
                           """)

            rows = cursor.fetchall()
            # 테이블별로 그룹화
            tables = defaultdict(list)
            for table_name, column_name, data_type, column_comment in rows:
                tables[table_name].append({
                    "column_name": column_name,
                    "data_type": data_type,
                    "comment": column_comment
                })

            result = [{"table_name": tbl, "columns": cols} for tbl, cols in tables.items()]
            return {"message": result, "status": "success"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()

@mcp.tool()
async def execute_postgres_query(query: str) -> dict:
    """사용자가 요청한 SQL 쿼리를 실행하는 도구입니다.
    - 쿼리가 명확히 주어진 경우 이 도구만 호출하면 됩니다.
    - 쿼리를 자동 생성해야 하는 경우, 필요하다면 먼저 fetch_table_schema를 호출해 스키마를 확인한 뒤 실행하세요.
    - 보안상 SELECT 쿼리만 허용됩니다."""
    try:
        # 데모에서는 읽기 전용 쿼리만 허용 (SQL Injection 완화)
        query_upper = query.strip().upper()
        if not query_upper.startswith('SELECT'):
            return {
                "message": "보안상 SELECT 쿼리만 허용됩니다.",
                "status": "error"
            }
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_session(readonly=True)  # 읽기 전용 모드
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        return {
            "message": "\n".join(str(row) for row in result),
            "status": "success"
        }
    except Exception as e:
        return {
            "message": str(e),
            "status": "error"
        }
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    mcp.run()
