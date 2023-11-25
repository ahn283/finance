import requests as rq
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO
from enum import Enum
import re
import pymysql
import time



# get sector data from KRX
class Get_sector_data():
        
    def __init__(self) -> None:
        self.BIZ_DAY_URL = 'https://finance.naver.com/sise/sise_deposit.nhn'
        self.SECTOR_OTP_URL = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
        self.SECTOR_OTP_REFERER_URL = 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader'
        self.SECTOR_DOWNLOAD_URL = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
        self.SECTOR_JSON_URL = 'dbms/MDC/STAT/standard/MDCSTAT03901'
        self.SECTOR_JSON_URL_IND = 'dbms/MDC/STAT/standard/MDCSTAT03501'
        self.biz_day = self.get_biz_day()
        
    def update_database(self):
        con = pymysql.connect(user='root',
                              passwd='04250629',
                              host='127.0.0.1',
                              db='stock',
                              charset='utf8')
        mycursor = con.cursor()
        query = f"""
        INSERT INTO ticker_kr (index_code, company_code, company, sec_nm_kr, date)
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
        index_code=VALUES(index_code), company_code=VALUES(company_code), company=VALUES(company), sec_nm_kr=VALUES(sec_nm_kr)
        """
        
        kor_sector = self.get_sector_data()

        args = kor_sector.values.tolist()
        print(args)

        mycursor.executemany(query, args)
        con.commit()

        con.close()
        
    
    def get_sector_data(self):
        krx_sector = self.get_dataframe()
        krx_ind = self.get_dataframe_ind()
        diff = list(set(krx_sector['종목명']).symmetric_difference(set(krx_ind['종목명'])))
        print(diff)
        kor_ticker = pd.merge(
            krx_sector,
            krx_ind,
            on=krx_sector.columns.intersection(
                krx_ind.columns
            ).to_list(), how='outer'
        )
        
        # 종목 구분
        kor_ticker['종목구분'] = np.where(kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'), '스팩',
                                     np.where(kor_ticker['종목코드'].str[-1:] != '0', '우선주',
                                              np.where(kor_ticker['종목명'].str.endswith('리츠'), '리츠',
                                                       np.where(kor_ticker['종목명'].isin(diff), '기타', '보통주'))))
        kor_ticker = kor_ticker.reset_index(drop=True)
        kor_ticker.columns = kor_ticker.columns.str.replace(' ', '')        # 열이름 공백 제거
        kor_ticker = kor_ticker[['종목코드', '종목명', '시장구분', '종가',
                                 '시가총액', '기준일', 'EPS', '선행EPS', 'BPS', '주당배당금', '종목구분']]
        kor_ticker = kor_ticker.replace({np.nan : None})       # SQL에는 NaN이 입력되지 않으므로 None으로 변경
        # kor_ticker['기준일'] = pd.to_datetime(kor_ticker['기준일'])
        
        return kor_ticker 
    
    
    # krx ind
    def get_dataframe_ind(self):
        krx_ind = self.download_data(exchange='ALL')
        krx_ind['종목명'] = krx_ind['종목명'].str.strip()
        krx_ind['기준일'] = self.biz_day
        
        return krx_ind

             
      
    # concat kospi and kosdaq sector codes  
    def get_dataframe(self):
        sector_stk = self.download_data(exchange='KOSPI')
        sector_ksq = self.download_data(exchange='KOSDAQ')
        krx_sector = pd.concat([sector_stk, sector_ksq]).reset_index(drop=True)
        # eliminate blank name with strip() method
        krx_sector['종목명'] = krx_sector['종목명'].str.strip()
        # add '기준일' column
        krx_sector['기준일'] = self.biz_day
        return krx_sector
    
    # get recent business day
    def get_biz_day(self):
        url = self.BIZ_DAY_URL
        data = rq.get(url)
        data_html = BeautifulSoup(data.content)
        parse_day = data_html.select_one('div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah').text
        
        biz_day = re.findall('[0-9]+', parse_day)
        biz_day = ''.join(biz_day)
        
        return biz_day
    
    def get_otp(self, market, url):
        gen_otp_url = self.SECTOR_OTP_URL
        gen_otp = {
            'mktId' : market,        # STK는 코스피
            'trdDd' : self.biz_day,
            'money' : '1',
            'csvxls_isNo' : 'false',
            'name' : 'fileDown',
            'url' : url
        }
        print(gen_otp_url)
        print(gen_otp)
        # 헤더 부분에 레퍼러 추가 : 첫번째 URL에서 OTP를 부여받고, 이를 다시 두번째 URL에 제공하는 과정에서 레퍼러 없이 OTP를 전달하면 봇으로 인식해 데이터를 주지 않는다.
        headers = {'Referer': self.SECTOR_OTP_REFERER_URL}
        # post() 함수를 통해 해당 URL에 쿼리를 전송하면 이에 해당하는 데이터를 받으며, 이 중에 텍스트에 해당하는 내용만 불러온다.
        otp_stk = rq.post(gen_otp_url, gen_otp, headers=headers).text
        
        return otp_stk, headers
    
    def download_data(self, exchange):
        time.sleep(2)
        down_url = self.SECTOR_DOWNLOAD_URL
        if exchange == 'KOSPI':
            market = 'STK'
            url = self.SECTOR_JSON_URL
        elif exchange == 'KOSDAQ':
            market = 'KSQ'
            url = self.SECTOR_JSON_URL
        else:
            market = 'ALL'
            url = self.SECTOR_JSON_URL_IND
        otp, headers = self.get_otp(market=market, url=url)
        
        print(market, url)
        # post otp and download data
        down_sector = rq.post(down_url, {'code': otp}, headers=headers)
        # change data's content into binary stream using ByteIO() 
        # read it with read_csv() function
        sector = pd.read_csv(BytesIO(down_sector.content), encoding='EUC-KR')
        
        print(sector)
        
        return sector
    
if  __name__ == '__main':
    sector = Get_sector_data()
    sector.update_database()
        
        