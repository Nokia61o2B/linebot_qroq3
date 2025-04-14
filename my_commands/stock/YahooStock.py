import requests
from bs4 import BeautifulSoup

class YahooStock:
    """
    Yahoo 股市資訊類別，透過股票代號取得股票名稱、價格、漲跌幅、幣別和收盤時間資訊。
    """

    def __init__(self, stock_symbol: str):
        """
        初始化 YahooStock 實例，並抓取初始值。

        參數:
            stock_symbol (str): 股票代號，例如 '2330' 或 'qqq'。
        """
        self.stock_symbol = stock_symbol
        self.name = None
        self.now_price = None
        self.change = None
        self.currency = None
        self.close_time = None  # 新增屬性：完整收盤時間文字

        # 初始化時自動抓取資訊
        try:
            stock_info = self.fetch_stock_info()
            self.name = stock_info.get("name")
            self.now_price = stock_info.get("now_price")
            self.change = stock_info.get("change")
            self.currency = stock_info.get("currency")
            self.close_time = stock_info.get("close_time")
        except ValueError as e:
            print(f"初始化失敗: {str(e)}")

    def fetch_stock_info(self):
        """
        從 Yahoo 股市抓取股票資訊並更新屬性。

        傳回:
            dict: 包含 'name' (股票名稱), 'now_price' (目前價格),
                  'change' (漲跌幅), 'currency' (幣別), 'close_time' (完整收盤時間文字) 的字典。
        """
        base_url = f'https://tw.stock.yahoo.com/quote/{self.stock_symbol}'

        try:
            # 取得網頁內容
            response = requests.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            checkus = soup.find('span', class_='Jc(fe) Fz(20px) Lh(1.2) Fw(b) D(f) Ai(c) C($c-trend-up)')
            if not checkus:
                # 台股邏輯
                self.name = soup.find('h1', class_='C($c-link-text) Fw(b) Fz(24px) Mend(8px)').get_text(strip=True)
                price_element = soup.find('span', class_='Fz(32px)')
                self.now_price = price_element.get_text(strip=True)
                change_element = soup.find('span', class_='Fz(20px)')
                change_value = change_element.get_text(strip=True) if change_element else ""
                trend = ''
                if 'C($c-trend-down)' in change_element['class']:
                    trend = '-'
                elif 'C($c-trend-up)' in change_element['class']:
                    trend = '+'
                self.change = f"{trend}{change_value}"
                self.currency = "TWD"  # 預設台股幣別

                # 抓取台股收盤時間
                close_time_element = soup.find('span', class_='C(#6e7780) Fz(12px) Fw(b)')
                self.close_time = close_time_element.get_text(strip=True) if close_time_element else None
            else:
                # 美股邏輯
                self.name = soup.find('h1', class_='C($c-link-text) Fw(b) Fz(24px) Mend(8px)').get_text(strip=True)
                price_element = soup.find('span', class_='Fz(32px)')
                self.now_price = price_element.get_text(strip=True)
                change_element_up = soup.find('span', class_='Fz(20px) Fw(b) Lh(1.2) Mend(4px) D(f) Ai(c) C($c-trend-up)')
                change_element_down = soup.find('span', class_='Fz(20px) Fw(b) Lh(1.2) Mend(4px) D(f) Ai(c) C($c-trend-down)')
                change_element = change_element_up or change_element_down
                if change_element:
                    change_value = change_element.get_text(strip=True)
                    trend = ''
                    if 'C($c-trend-down)' in change_element['class']:
                        trend = '-'
                    elif 'C($c-trend-up)' in change_element['class']:
                        trend = '+'
                    self.change = f"{trend}{change_value}"
                else:
                    self.change = None
                currency_element = soup.find('span', class_='Fz(20px)')
                currency_value = currency_element.get_text(strip=True) if currency_element else ""
                self.currency = currency_value

                # 抓取美股收盤時間
                close_time_element = soup.find('span', class_='C(#6e7780) Fz(12px) Fw(b) Fw(400)!')
                self.close_time = close_time_element.get_text(strip=True) if close_time_element else None

            # 返回資訊
            return {
                "name": self.name,
                "now_price": self.now_price,
                "change": self.change,
                "currency": self.currency,
                "close_time": self.close_time  # 包含完整收盤時間文字
            }

        except Exception as e:
            raise ValueError(f"無法取得股票資訊: {str(e)}")