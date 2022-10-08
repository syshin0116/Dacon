#�������� : 2018.08.17
#������: �����

import pandas
from bs4 import BeautifulSoup
import requests
import datetime

#�ʹ� ���Ƽ� ����� ���� �Լ�
def find(a):
    f=soup.find(a)
    return f
def allfind(b):
    t=soup.find_all(b)
    return t

startnum=1
endnum=1000

#������ ��,�� ȹ��
d=datetime.date.today()
year=d.year
month=d.month
day=d.day
strday=str(day)

#���� ���ڸ� �� �� ��� 08�� ǥ���ǵ���
if day<10:
    strday='0'+str(day)

#������ ���� �ֽ� ������ ã��
while True:
    stryear = str(year)
    strmonth = str(month)
    #url �� ���� ���ڸ� �� �ϰ�� 08�� ǥ���ǵ���
    if month<10:
        strmonth='0'+strmonth
    url='http://openapi.seoul.go.kr:8088/6d4b4f69576c756b333544584a7942/xml/InfoTrdarFlpop/1/1000/'+stryear+strmonth
    req=requests.get(url)
    html=req.text
    soup=BeautifulSoup(html, 'html.parser')

    #���ο��� Ȯ��
    reqcode = str(soup.find('message'))
    #���ν� break
    if reqcode=='<message>���� ó���Ǿ����ϴ�</message>':
        break
    #���� �����ʾ��� ��� month�� �ϳ���
    month=month-1
    #month�� 0�̵� ��� year�� �ϳ� ���� month�� 12�� �ǵ���
    if month==0:
        year=year-1
        month=12

#����Ʈ�� 1-1000���� �ڷ� ����
codenumber = allfind('trdar_cd') #���θ� ���� �ѹ�
roadname = allfind('trdar_cd_nm') #���θ�
totalnumber = allfind('tot_flpop_co') #�ش� ������ �� �����α� ��
maletotal = allfind('ml_flpop_co') #�ش� ������ �� ���� �����α� ��
femaletotal = allfind('fml_flpop_co') #�ش� ������ �� ���� �����α� ��
age_10 = allfind('agrde_10_flpop_co') #�ش� ������ 10�� �����α� ��
age_20 = allfind('agrde_20_flpop_co') #�ش� ������ 20�� �����α� ��
age_30 = allfind('agrde_30_flpop_co') #�ش� ������ 30�� �����α� ��
age_40 = allfind('agrde_40_flpop_co') #�ش� ������ 40�� �����α� ��
age_50 = allfind('agrde_50_flpop_co') #�ش� ������ 50�� �����α� ��
age_60 = allfind('agrde_60_above_flpop_co') #�ش� ������ 60�� �̻� �����α� ��

#����� ��å�� �ѹ��� 1000������ ��û ���� 1001-2000���� ����
url='http://openapi.seoul.go.kr:8088/6d4b4f69576c756b333544584a7942/xml/InfoTrdarFlpop/1001/2000/'+stryear+strmonth
req = requests.get(url)
html = req.text
soup = BeautifulSoup(html, 'html.parser')
codenumber_2 = allfind('trdar_cd') #���θ� ���� �ѹ�
roadname_2 = allfind('trdar_cd_nm') #���θ�
totalnumber_2 = allfind('tot_flpop_co') #�ش� ������ �� �����α� ��
maletotal_2 = allfind('ml_flpop_co') #�ش� ������ �� ���� �����α� ��
femaletotal_2 = allfind('fml_flpop_co') #�ش� ������ �� ���� �����α� ��
age_10_2 = allfind('agrde_10_flpop_co') #�ش� ������ 10�� �����α� ��
age_20_2 = allfind('agrde_20_flpop_co') #�ش� ������ 20�� �����α� ��
age_30_2 = allfind('agrde_30_flpop_co') #�ش� ������ 30�� �����α� ��
age_40_2 = allfind('agrde_40_flpop_co') #�ش� ������ 40�� �����α� ��
age_50_2 = allfind('agrde_50_flpop_co') #�ش� ������ 50�� �����α� ��
age_60_2 = allfind('agrde_60_above_flpop_co') #�ش� ������ 60�� �̻� �����α� ��

#����Ʈ �ΰ� �ջ�
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

#csv���Ϸ� ����
number=0
today=stryear+strmonth+strday
with open(today+'_�������.csv','w') as file:
    file.write('�����ڵ�,���θ�,��ü �����α�,���� �����α�,���� �����α�,10�� �����α�,20�� �����α�,30�� �����α�,'
               '40�� �����α�, 50�� �����α�, 60�� �̻� �����α�\n')
    for i in codenumber:
        file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n'.format(codenumber[number].text,roadname[number].text,totalnumber[number].text,
                                                                           maletotal[number].text,femaletotal[number].text,age_10[number].text,
                                                                           age_20[number].text,age_30[number].text,age_40[number].text,age_50[number].text,
                                                                           age_60[number].text))
        number=number+1
    file.close()