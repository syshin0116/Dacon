# Pandas 실습

import pandas as pd
import numpy as np

# 생성하기
a = pd.DataFrame([[1,2],[2,3]],columns = ['col1', 'col2'])
print(a)

# 생성하기 csv
path = "/Users/syshin/Desktop/Syshin/Dacon/서울 시민 생활 데이터를 활용한 도시문제 해결 경진대회/1인가구.csv"
data = pd.read_csv(path, encoding='cp949')

# 생성하기 csv
#data = pd.read_excel('1인가구.xlsx', 0)

# 열이름 변경하기
a.columns[1] = 'col2-1' #  error
a.rename( columns = {a.columns[1]: 'col2-1'})
a
# Tips! inplace 
a.rename( columns = {a.columns[1]: 'col2-1'}, inplace = True)
a


## rename 을 쓰지 않고 컬럼명 바꾸는 방법
acol = list(a.columns)
acol[1] = 'col2-1'
a.columns = acol
a

# 참조 및 추출
a.iloc[0,:]
a.iloc[0,:] = [10,20]
a

a.loc[:, [True, False]]
a.iloc[:, [0]]


# 데이터의  추가
a.append([[1,2]])
# why? 열 이름을 이용하여 병합하기 때문임 

# note 1
pd.concat([ a, pd.DataFrame([[1,2]]) ], axis = 0)
# note 2
b = pd.DataFrame([[10,20]])
b.columns = a.columns
b
pd.concat([a, b], axis = 0)
# note 2-1
b = pd.DataFrame([[10,20]], columns = a.columns)
pd.concat([a, b], axis = 0)

# 실패한 append 를 다음과 같이 병합
a.append( pd.DataFrame([[10,20]], columns = a.columns))

# 다음과 같은 경우에 편리함. 
b =  pd.DataFrame([[20,10]], columns = ['col2', 'col1'])
b
pd.concat([a, b], axis = 0)

# random 
amat = np.random.normal(0,1, size  = (10,2))
amat_pdf = pd.DataFrame(amat)
v = []
for i in range(2):
    v.append("v"+ str(i))
v
amat_pdf.columns = v

# iterable객체 활용 tip
["v"+str(i) for i in range(2)]

# 함수
data.columns
data.iloc[:,0].unique()
data['자치구'].value_counts()
data['1인가구수'].sum()
