import requests

def get_btc_rate(currency='usd'):
    """
    獲取指定貨幣的當前比特幣匯率。

    :param currency: 要獲取比特幣匯率的貨幣（例如 'usd', 'twd'）。
    :return: 指定貨幣的當前比特幣匯率，如果發生錯誤則返回 None。
    """
    try:
        # 用於從 CoinGecko API 獲取指定貨幣的 BTC 價格的 URL
        url = f'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies={currency}'

        # 發送 GET 請求到 API
        response = requests.get(url)

        # 檢查請求是否成功（狀態碼 200）
        if response.status_code == 200:
            # 解析 JSON 響應
            data = response.json()

            # 從響應中提取 BTC 價格
            btc_price = data['bitcoin'][currency]

            return btc_price
        else:
            # 如果請求失敗，打印錯誤訊息
            print("無法獲取比特幣匯率。狀態碼:", response.status_code)
            return None
    except Exception as e:
        # 打印過程中出現的任何異常
        print("發生錯誤:", str(e))
        return None

"""
# 示例用法：
if __name__ == '__main__':
    btc_rate_usd = get_btc_rate('usd')
    if btc_rate_usd is not None:
        print("當前比特幣匯率（美元）: $", btc_rate_usd)

    btc_rate_twd = get_btc_rate('twd')
    if btc_rate_twd is not None:
        print("當前比特幣匯率（台幣）: $", btc_rate_twd)
"""