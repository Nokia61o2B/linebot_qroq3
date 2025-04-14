import requests
from bs4 import BeautifulSoup

class CaiyunfangweiCrawler:
    def __init__(self, url="https://calendar.8s8s.net/caiyunfangwei.php"):
        self.url = url

    def get_caiyunfangwei(self):
        # 發送GET請求到目標網址
        response = requests.get(self.url)

        # 檢查請求是否成功
        if response.status_code == 200:
            # 使用BeautifulSoup解析網頁內容
            soup = BeautifulSoup(response.content, 'html.parser')

            # 查找 class 為 'cd3_text' 的 div
            cd3_text_div = soup.find('div', class_='cd3_text')

            if cd3_text_div:
                # 初始化變數
                today_date = ''
                sui_ci = ''
                cai_shen_fang_wei = ''

                # 在該 div 內查找所有的 p 標籤
                p_tags = cd3_text_div.find_all('p')

                # 遍歷所有的 p 標籤，查找相應的信息
                for p in p_tags:
                    span_tag = p.find('span')
                    if span_tag:
                        span_text = span_tag.get_text()
                        if '今天日期：' in span_text:
                            today_date = p.find('strong').get_text()
                        elif '今日歲次：' in span_text:
                            sui_ci = p.contents[1].strip()
                        elif '財神方位：' in span_text:
                            cai_shen_fang_wei = p.find('font').get_text()

                return {
                    "今天日期": today_date,
                    "今日歲次": sui_ci,
                    "財神方位": cai_shen_fang_wei
                }
            else:
                raise ValueError("找不到 class 為 'cd3_text' 的 div 元素。")
        else:
            raise ConnectionError(f"無法訪問網站，狀態碼：{response.status_code}")
