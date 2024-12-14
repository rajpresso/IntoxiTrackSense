CREATE TABLE csv_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL, -- 파일명 (확장자 제외)
    version VARCHAR(50) NOT NULL,    -- 버전
    category VARCHAR(50) NOT NULL,    -- 버전
    sub_category VARCHAR(50) ,    -- 버전
    data JSON,                       -- CSV 데이터를 JSON 형태로 저장
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 자동 등록 날짜
    UNIQUE(file_name, version)       -- 파일명과 버전 조합에 대해 고유 제약조건
);