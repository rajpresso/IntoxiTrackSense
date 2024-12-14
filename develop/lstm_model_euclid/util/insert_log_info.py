import os
import mysql.connector
import pandas as pd
import json
from config.db_cfg import DB_CONFIG


def insert_log( date_info, moniter_no, log_content, file_url):

    db_config = {
    'user': 'alpaco',
    'password': '1234',
    'host': '112.175.29.231',
    'database': 'drunk',
    'port': 3306  # MySQL 기본 포트
    }
    try:
        # MySQL 데이터베이스 연결
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        date_info, moniter_no, log_content, file_url
        # date_info, moniter_no, log_content, file_url
                    # 데이터 삽입
        try:
            insert_query = """
                INSERT INTO log_info (date_info, moniter_no, log_content, file_url)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                date_info,
                moniter_no,
                log_content,
                file_url
            ))
            conn.commit()
            print(f"Logs have been saved to the database.")
        except Exception as e:
            print(f"Error saving Log to the database: {e}")
            conn.rollback()
            


    except mysql.connector.Error as db_error:
        print(f"Database connection error: {db_error}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
