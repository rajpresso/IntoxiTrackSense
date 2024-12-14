import os
import mysql.connector
import pandas as pd
import json
from config.db_cfg import DB_CONFIG



def insert_data( folder_path, version, category, sub_category):

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

        # 폴더 내 CSV 파일 목록 가져오기
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for file in files:
            file_path = os.path.join(folder_path, file)
            file_name = os.path.splitext(file)[0]  # 확장자 제외 파일명
            
            print(f"file_name {file_name} major_category {category}")

            # CSV 파일 읽기
            try:
                df = pd.read_csv(file_path)
                df['label'] = df['label'].astype(str).str.replace("[: ]", "", regex=True)
                df['filename'] = file_name
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            # 테이블 컬럼 수와 비교
            # table_columns = 39  # 예: 테이블에 저장될 데이터의 컬럼 수
            # if len(df.columns) != table_columns:
            #     print(f"Column mismatch in {file}. Expected {table_columns}, but got {len(df.columns)}.")
            #     continue

            # 중복 확인
            cursor.execute(
                "SELECT 1 FROM csv_data WHERE file_name = %s AND version = %s",
                (file_name, version)
            )
            if cursor.fetchone():
                print(f"File '{file}' with version '{version}' already exists in the database.")
                continue

            # 데이터 삽입
            try:
                insert_query = """
                    INSERT INTO csv_data (file_name, version, category,sub_category,data)
                    VALUES (%s, %s, %s, %s, %s)
                """
                df.fillna(value=0, inplace=True)  
                cursor.execute(insert_query, (
                    file_name,
                    version,
                    category,
                    sub_category,
                    json.dumps(df.to_dict(orient='records'))  # 데이터를 JSON 형식으로 저장
                ))
                conn.commit()
                print(f"File '{file}' successfully stored in the database.")
            except Exception as e:
                print(f"Error saving {file} to the database: {e}")
                conn.rollback()

    except mysql.connector.Error as db_error:
        print(f"Database connection error: {db_error}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# def import_data( version, category ):
#     """
#     MySQL 데이터베이스에서 주어진 버전의 JSON 데이터를 읽어와 pandas DataFrame으로 반환.
    
#     Args:
#         version (str): 검색할 데이터의 버전.
        
#     Returns:
#         pd.DataFrame: 검색된 데이터를 포함한 pandas DataFrame.
#     """
# # 데이터베이스 연결 정보
#     db_config = DB_CONFIG
#     # db_config = {
#     # 'user': 'alpaco',
#     # 'password': '1234',
#     # 'host': '112.175.29.231',
#     # 'database': 'drunk',
#     # 'port': 3306  # MySQL 기본 포트
#     # }
#     try:
#         # MySQL 데이터베이스 연결
#         conn = mysql.connector.connect(**db_config)
#         cursor = conn.cursor()

#         # JSON 데이터를 읽어오는 SQL 쿼리
#         query = """
#             SELECT data FROM csv_data WHERE version = %s
#         """
#         cursor.execute(query, (version,))
#         result = cursor.fetchall()

#         # 데이터가 없는 경우 처리
#         if not result:
#             print(f"No data found for version '{version}'")
#             return pd.DataFrame()  # 빈 DataFrame 반환

#         # JSON 데이터를 DataFrame으로 변환
#         # 각 행에서 JSON 데이터를 파싱하고 합칩니다.
#         all_data = []
#         for (json_data,) in result:
#             # JSON 문자열을 파이썬 객체로 변환
#             rows = json.loads(json_data)
#             all_data.extend(rows)  # 각 JSON 레코드를 리스트에 추가

#         # pandas DataFrame 생성
#         df = pd.DataFrame(all_data)
#         return df

#     except mysql.connector.Error as db_error:
#         print(f"Database connection error: {db_error}")
#         return pd.DataFrame()  # 빈 DataFrame 반환

#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'conn' in locals():
#             conn.close()