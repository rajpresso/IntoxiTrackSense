import pandas as pd
import numpy as np
import os
import mysql.connector
import pandas as pd
import json
from config.db_cfg import DB_CONFIG


def import_df(): 
    Relative_Dataframe = pd.DataFrame() 
    for folders in os.listdir('/home/alpaco/project/drunk_prj/data/normal_ver2'):
        new_path = os.path.join('/home/alpaco/project/drunk_prj/data/normal_ver2',folders)
        new_path = os.path.join(new_path,'rel')
        for csvf in os.listdir(new_path):
            new_csv = os.path.join(new_path,csvf)
            print("et")
            temp_df = pd.read_csv(new_csv)
            temp_df['FILENAME'] = (csvf.split('/')[-1]).split('.')[0]
            num_cols = temp_df.select_dtypes(include=['number']).columns  # 숫자형 열만 선택
            temp_df[num_cols] = temp_df[num_cols].clip(lower=0)
            Relative_Dataframe = pd.concat([Relative_Dataframe,temp_df],ignore_index=True)
            print(new_csv)
    Relative_Dataframe['y']=0
    Relative_Dataframe

    Relative_Dataframe2 = pd.DataFrame()
    idx=0
    for folder in os.listdir('/home/alpaco/project/drunk_prj/data/comfirm_video1/totter/Rel'):
        rel = os.path.join('/home/alpaco/project/drunk_prj/data/comfirm_video1/totter/Rel',folder)
        for csvf in os.listdir(rel):
            new_path = os.path.join(rel,csvf)
            temp_df = pd.read_csv(new_path)
            temp_df['FILENAME'] = (csvf.split('/')[-1]).split('.')[0]
            num_cols = temp_df.select_dtypes(include=['number']).columns  # 숫자형 열만 선택
            temp_df[num_cols] = temp_df[num_cols].clip(lower=0)
            temp_df = temp_df.iloc[::10, :]
            Relative_Dataframe2 = pd.concat([Relative_Dataframe2,temp_df],ignore_index=True)
    Relative_Dataframe2['y']=1
    Relative_Dataframe2

    Combined = pd.concat([Relative_Dataframe,Relative_Dataframe2],ignore_index=True)
    return Combined





def import_data(version, category='', sub_category=''):
    # 데이터베이스 연결 정보
    db_config = DB_CONFIG
    try:
        # MySQL 데이터베이스 연결
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 기본 SQL 쿼리
        query = "SELECT data FROM csv_data WHERE version = %s"
        params = [version]

        # category와 sub_category 조건 추가
        if category:
            query += " AND category = %s"
            params.append(category)
        if sub_category:
            query += " AND sub_category = %s"
            params.append(sub_category)

        print(f'query: {query} ')

        # 쿼리 실행
        cursor.execute(query, params)
        result = cursor.fetchall()

        # 데이터가 없는 경우 처리
        if not result:
            print(f"No data found for version '{version}', category '{category}', and sub_category '{sub_category}'")
            return pd.DataFrame()  # 빈 DataFrame 반환

        # JSON 데이터를 DataFrame으로 변환
        all_data = []
        for (json_data,) in result:
            # JSON 문자열을 파이썬 객체로 변환
            rows = json.loads(json_data)
            all_data.extend(rows)  # 각 JSON 레코드를 리스트에 추가

        # pandas DataFrame 생성
        df = pd.DataFrame(all_data)
        return df

    except mysql.connector.Error as db_error:
        print(f"Database connection error: {db_error}")
        return pd.DataFrame()  # 빈 DataFrame 반환

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

    