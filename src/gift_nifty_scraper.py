import os, json, re, requests
from datetime import datetime, timezone

LIVEINDEX_URL = "https://liveindex.org/futures/gift-nifty-futures/"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CACHE_FILE = os.path.join(CACHE_DIR, "gift_nifty_cache.json")

_REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def parse_liveindex_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")
    cells = rows[1].find_all("td")
    price = parse_num(cells[1].get_text(strip=True))
    change_pts = parse_num(cells[2].get_text(strip=True))
    change_pct_str = cells[3].get_text(strip=True)
    m = re.search(r"([+-]?\d+\.?\d*)", change_pct_str)
    change_pct = float(m.group(1)) if m else None
    prev_close = round(price - change_pts, 1) if price is not None and change_pts is not None else None
    return price, change_pct, prev_close

def parse_num(s):
    m = re.search(r"([+-]?[\d,]+\.?\d*)", s)
    return float(m.group(1).replace(",", "")) if m else None

def scrape_gift_nifty(timeout_sec=20):
    try:
        r = requests.get(LIVEINDEX_URL, headers=_REQ_HEADERS, timeout=timeout_sec)
        r.raise_for_status()
        price, change_pct, prev_close = parse_liveindex_html(r.text)
        if price is None or change_pct is None:
            return None, "parse failed"
        result = {"price": price, "change_pct": change_pct, "prev_close": prev_close,
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(result, f)
        return result, None
    except Exception as e:
        return None, str(e)

def get_gift_nifty_gap(use_cache=True, max_age_hours=1):
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                data = json.load(f)
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(data["timestamp"])).total_seconds() / 3600
            if age <= max_age_hours:
                return data["change_pct"], data
        except:
            pass
    result, err = scrape_gift_nifty()
    if result is not None:
        return result["change_pct"], result
    return None, err

if __name__ == "__main__":
    r, e = get_gift_nifty_gap(use_cache=False)
    if r is not None:
        print(f"GIFT Nifty gap: {r:+.2f}%  Price: {e['price']}  Prev: {e['prev_close']}")
    else:
        print(f"Failed: {e}")
