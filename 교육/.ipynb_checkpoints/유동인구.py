#개발일자 : 2018.08.17
#개발자: 신재욱

import pandas
from bs4 import BeautifulSoup
import requests
import datetime

#너무 많아서 사용자 지정 함수
def find(a):
    f=soup.find(a)
    return f
def allfind(b):
    t=soup.find_all(b)
    return t

startnum=1
endnum=1000

#오늘의 연,월 획득
d=datetime.date.today()
year=d.year
month=d.month
day=d.day
strday=str(day)

#일이 한자리 수 일 경우 08로 표현되도록
if day<10:
    strday='0'+str(day)

#가능한 가장 최신 데이터 찾기
while True:
    stryear = str(year)
    strmonth = str(month)
    #url 뒤 월이 한자리 수 일경우 08로 표현되도록
    if month<10:
        strmonth='0'+strmonth
    url='http://openapi.seoul.go.kr:8088/6d4b4f69576c756b333544584a7942/xml/InfoTrdarFlpop/1/1000/'+stryear+strmonth
    req=requests.get(url)
    html=req.text
    soup=BeautifulSoup(html, 'html.parser')

    #승인여부 확인
    reqcode = str(soup.find('message'))
    #승인시 break
    if reqcode=='<message>정상 처리되었습니다</message>':
        break
    #승인 되지않았을 경우 month를 하나뺌
    month=month-1
    #month가 0이될 경우 year을 하나 빼고 month를 12로 되돌림
    if month==0:
        year=year-1
        month=12

#리스트에 1-1000까지 자료 저장
codenumber = allfind('trdar_cd') #도로명 고유 넘버
roadname = allfind('trdar_cd_nm') #도로명
totalnumber = allfind('tot_flpop_co') #해당 도로의 총 유동인구 수
maletotal = allfind('ml_flpop_co') #해당 도로의 총 남성 유동인구 수
femaletotal = allfind('fml_flpop_co') #해당 도로의 총 여성 유동인구 수
age_10 = allfind('agrde_10_flpop_co') #해당 도로의 10대 유동인구 수
age_20 = allfind('agrde_20_flpop_co') #해당 도로의 20대 유동인구 수
age_30 = allfind('agrde_30_flpop_co') #해당 도로의 30대 유동인구 수
age_40 = allfind('agrde_40_flpop_co') #해당 도로의 40대 유동인구 수
age_50 = allfind('agrde_50_flpop_co') #해당 도로의 50대 유동인구 수
age_60 = allfind('agrde_60_above_flpop_co') #해당 도로의 60대 이상 유동인구 수

#서울시 정책상 한번에 1000번까지 요청 가능 1001-2000까지 저장
url='http://openapi.seoul.go.kr:8088/6d4b4f69576c756b333544584a7942/xml/InfoTrdarFlpop/1001/2000/'+stryear+strmonth
req = requests.get(url)
html = req.text
soup = BeautifulSoup(html, 'html.parser')
codenumber_2 = allfind('trdar_cd') #도로명 고유 넘버
roadname_2 = allfind('trdar_cd_nm') #도로명
totalnumber_2 = allfind('tot_flpop_co') #해당 도로의 총 유동인구 수
maletotal_2 = allfind('ml_flpop_co') #해당 도로의 총 남성 유동인구 수
femaletotal_2 = allfind('fml_flpop_co') #해당 도로의 총 여성 유동인구 수
age_10_2 = allfind('agrde_10_flpop_co') #해당 도로의 10대 유동인구 수
age_20_2 = allfind('agrde_20_flpop_co') #해당 도로의 20대 유동인구 수
age_30_2 = allfind('agrde_30_flpop_co') #해당 도로의 30대 유동인구 수
age_40_2 = allfind('agrde_40_flpop_co') #해당 도로의 40대 유동인구 수
age_50_2 = allfind('agrde_50_flpop_co') #해당 도로의 50대 유동인구 수
age_60_2 = allfind('agrde_60_above_flpop_co') #해당 도로의 60대 이상 유동인구 수

#리스트 두개 합산
codenumber = codenumber+codenumber_2
roadname = roadname+roadname_2
totalnumber = totalnumber+totalnumber_2
maletotal = maletotal + maletotal_2
femaletotal = femaletotal+femaletotal_2
age_10 = age_10+age_10_2
age_20 = age_20+age_20_2
age_30 = age_30+age_30_2
age_40 = age_40+age_40_2
age_50 = age_50+age_50_2
age_60 = age_60+age_60_2

#csv파일로 저장
number=0
today=stryear+strmonth+strday
with open(today+'_상권정보.csv','w') as file:
    file.write('도로코드,도로명,전체 유동인구,남성 유동인구,여성 유동인구,10대 유동인구,20대 유동인구,30대 유동인구,'
               '40대 유동인구, 50대 유동인구, 60대 이상 유동인구\n')
    for i in codenumber:
        file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n'.format(codenumber[number].text,roadname[number].text,totalnumber[number].text,
                                                                           maletotal[number].text,femaletotal[number].text,age_10[number].text,
                                                                           age_20[number].text,age_30[number].text,age_40[number].text,age_50[number].text,
                                                                           age_60[number].text))
        number=number+1
    file.close()