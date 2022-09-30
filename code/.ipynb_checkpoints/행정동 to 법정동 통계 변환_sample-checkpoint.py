#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np

# 행정동과 법정동을 매칭시켜주는 파일을 불러옴
df_match = pd.read_csv('D:/GIS/statistics/KIKmix.20200814.csv', \
                       encoding='utf-8')

# 서울시, 종로구 등 시군구 단위까지만 있는 것은
# 값이 없기 때문에 결측치로 처리됨 : 해당 결측치 제거
df_match = df_match.dropna()

# 행정동/법정동 코드는 현행 8자리이나, 
# 행안부 홈페이지에서 받은 자료는 10자리로 되어 있음
# 현행에 맞게 변경 (미래를 위해서 해둔건가?)
df_match['행정동코드'] = (df_match['행정동코드']/100).astype(np.int64)
df_match['법정동코드'] = (df_match['법정동코드']/100).astype(np.int64)

# 하나의 행정동이 몇개의 법정동으로 이루어져 있는지를 추가.
# 10이면 10개의 법정동이 하나의 행정동을 이루고 있는 것
df_match = pd.merge(df_match, df_match[['행정동코드','법정동코드']] \
	.groupby('행정동코드').count() \
	.rename(columns={'법정동코드':'법정동갯수'}).reset_index(), how='left')


# 추가하고자 하는 통계치 파일을 불러옴
# 불러오기 전에 파일을 먼저 확인해보면,
# 데이터 열 갯수보다 제목 열 갯수가 하나 적음.
# 이러면 pandas에서 이상하게 불러와지므로
# 제목 열을 마지막에 아무 이름이나 하나 추가
df_stat = pd.read_csv('D:/GIS/statistics/LOCAL_PEOPLE_DONG_202008.csv', \
                       encoding='utf-8', index_col=None)

# 아무 이름이나 추가한 마지막 열 삭제
df_stat = df_stat.iloc[:,:-1]

# 테스트코드이므로, 임시로 작은 DataFrame을 제작
df_small = df_stat.loc[(df_stat['기준일ID']==20200801) \
                       & (df_stat['시간대구분']==14),:]
df_small = df_small.iloc[:,2:]

# 행정동-법정동 매칭 정보와 통계치를 붙임 : PK = 행정동코드
df = pd.merge(df_match, df_small, how='left')
df = df.dropna()

# 붙인 테이블에서 통계치에 해당하는 열 범위를 리스트로 지정
# 통계치 조정 루프용
col = list(df.columns)[8:] 

# 행정동-법정동 매칭하면서 통계치를 같이 변동시켜주는 루프
# 행정동 통계치를 법정동 갯수로 나눔
# 하나의 행정동에 인구가 1000명이고 
# 이 행정동이 10개의 법정동으로 되어 있으면
# 1000/10을 해서 각 법정동에 해당하는 인구를 100명으로 추정
for c in col:
    df[c] = df[c]/df['법정동갯수']

del df['법정동갯수'] # 이녀석은 용도를 다했으니 버리고
# 법정동 한글 이름은 법정동 코드랑 같이 따로 뽑아둔다.
df_info = df[['법정동코드','시도명','시군구명','동리명']] \
            .groupby('법정동코드').first().reset_index() 

# 마지막으로 하나의 법정동이 여러개의 행정동으로
# 이루어진 경우도 있어서 법정동 코드 별로 값을 전부 더함
df = df.groupby('법정동코드').sum().reset_index()

# 위처럼 하면 법정동 한글명이 사라지기 때문에,
# 앞서 미리 뽑아두었던 법정동 한글명이랑 붙인다
df = pd.merge(df_info, df, how='left')
df = df.dropna()

# 마지막으로, 파일로 저장
df[['법정동코드','동리명','총생활인구수']] \
  .to_csv('D:/GIS/statistics/LOCAL_PEOPLE_법정동_202008.csv', \
           encoding='utf-8')