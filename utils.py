#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clutch.co crawler (reviews + social links) with anti-bot mitigation.

- curl_cffi Session + HTTP/2
- Rotate impersonate (chrome141 -> chrome120 -> safari17_0)
- Retry with exponential backoff + jitter on 403/429
- Random sleep between requests
- Graceful skip when blocked
- CLI: --limit, --out, --start-url

Usage:
    python crawl_clutch.py --limit 20 --out clutch_reviews.xlsx
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Optional, List
from urllib.parse import urljoin
from sklearn.model_selection import train_test_split
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Tuple, List, Dict, Any
import shutil, random, tempfile, json
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    # Fallback khi không cài tqdm
    def tqdm(x, **kwargs):
        return x
from concurrent.futures import ThreadPoolExecutor, as_completed
from curl_cffi import requests as creq


# =========================
# Config mặc định
# =========================
DEFAULT_START_URL = "https://clutch.co/it-services"

# Không tự set User-Agent nếu dùng impersonate
COMMON_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://clutch.co/",
    "Upgrade-Insecure-Requests": "1",
}

# Thứ tự thử impersonate khi bị chặn
IMP_CHOICES = ["chrome141", "chrome120", "safari17_0"]

# Khoảng ngủ ngẫu nhiên giữa các request (giây)
SLEEP_MIN = 6.5
SLEEP_MAX = 12.0

# =========================
# Session toàn cục
# =========================
SESS = creq.Session(
    headers=COMMON_HEADERS,
    timeout=30,
)


# =========================
# HTTP helper
# =========================
def http_get(url: str, max_retries: int = 4) -> Optional[creq.Response]:
    """
    GET với retry/backoff. Đổi impersonate nếu 403/429.
    Trả về Response 200 hoặc None nếu thất bại.
    """
    delay = 2.0
    for i in range(max_retries):
        imp = IMP_CHOICES[min(i, len(IMP_CHOICES) - 1)]
        try:
            r = SESS.get(url, impersonate=imp, allow_redirects=True)
            code = r.status_code
            if code == 200:
                return r
            if code in (403, 429):
                # Bị chặn / rate limit → backoff + jitter rồi thử tiếp
                time.sleep(delay + random.uniform(0.5, 1.8))
                delay *= 1.8
                continue
            # Lỗi khác: thử 1-2 lần nhẹ
            time.sleep(1.0)
        except Exception:
            time.sleep(delay)
            delay *= 1.8
    return None


# =========================
# Parsers
# =========================
def extract_review_data(soup: BeautifulSoup) -> dict:
    container = soup.find("div", class_="profile-review__data")
    if not container:
        return {}

    result: dict = {}
    li_items = container.select("ul.data--list > li.data--item")

    for li in li_items:
        tooltip_html = li.get("data-tooltip-content", "")
        tooltip_label = (
            BeautifulSoup(tooltip_html, "html.parser").get_text(strip=True)
            if tooltip_html
            else None
        )
        text_parts = [
            t.strip()
            for t in li.stripped_strings
            if not t.lower().startswith("show more")
        ]
        if not text_parts:
            continue

        if tooltip_label:
            result[tooltip_label] = " ".join(text_parts)
        else:
            result.setdefault("unknown", []).append(" ".join(text_parts))
    return result


def extract_reviewer_info(soup: BeautifulSoup) -> dict:
    container = soup.find("div", class_="profile-review__reviewer")
    if not container:
        return {}

    result: dict = {}

    # Tên reviewer
    name_tag = container.find("div", class_="reviewer_card--name")
    if name_tag:
        result["reviewer_name"] = name_tag.get_text(strip=True)

    # Chức vụ & công ty
    position_tag = container.find("div", class_="reviewer_position")
    if position_tag:
        text = position_tag.get_text(strip=True)
        result["reviewer_position_raw"] = text
        if "," in text:
            parts = [p.strip() for p in text.split(",", 1)]
            result["reviewer_role"] = parts[0]
            result["reviewer_company"] = parts[1]
        else:
            result["reviewer_role"] = text

    # Trạng thái verified
    verified_tag = container.find(
        "span", class_="profile-review__reviewer-verification-badge-title"
    )
    if verified_tag:
        result["verified_status"] = verified_tag.get_text(strip=True)

    # Industry, Location, Client size, Review type
    list_items = container.select("ul.reviewer_list > li.reviewer_list--item")
    for li in list_items:
        tooltip_html = li.get("data-tooltip-content", "")
        label = (
            BeautifulSoup(tooltip_html, "html.parser").get_text(strip=True)
            if tooltip_html
            else None
        )
        value_tag = li.find("span", class_="reviewer_list__details-title")
        value = value_tag.get_text(strip=True) if value_tag else None
        if label and value:
            result[label] = value

    return result


def extract_social_links(scope: BeautifulSoup) -> dict:
    """
    Lấy social links ở vùng 'scope' (thường là section#contact; nếu không có thì truyền cả trang).
    """
    links: dict = {}
    social_links = scope.select(
        "div.profile-social-media__wrap a.profile-social-media__link"
    )
    for a in social_links:
        label = a.get("data-type") or a.get_text(strip=True)
        href = a.get("href")
        if label and href:
            links[f"{label.lower()} Company Outsource"] = href
    return links


# =========================
# Scrapers
# =========================
def get_base_url_company(start_url: str) -> List[str]:
    """
    Lấy danh sách URL company profile từ trang directory.
    """
    ans: List[str] = []
    resp = http_get(start_url)
    if not resp:
        print("❌ Không lấy được trang list (403/timeout).", file=sys.stderr)
        return ans

    soup = BeautifulSoup(resp.text, "html.parser")
    providers_list = soup.find(id="providers__list")
    if not providers_list:
        print("❌ Không tìm thấy providers__list (có thể bị anti-bot).", file=sys.stderr)
        return ans

    for li in providers_list.find_all("li", class_="provider-list-item"):
        cta = li.find("div", class_="provider__cta-container")
        if not cta:
            continue
        a = cta.find(
            "a",
            class_="provider__cta-link sg-button-v2 sg-button-v2--secondary directory_profile",
        )
        if a and a.get("href"):
            ans.append(urljoin("https://clutch.co", a["href"]))
    return ans


def get_detail_information(company_url: str) -> pd.DataFrame:
    """
    Lấy bảng reviews + social links từ 1 trang company.
    """
    resp = http_get(company_url)
    if not resp:
        print(f"⚠️ Bỏ qua (403/timeout): {company_url}", file=sys.stderr)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")

    contact_scope = soup.find("section", id="contact") or soup  # fallback
    link_social = extract_social_links(contact_scope)

    reviews_wrap = soup.find("div", class_="profile-reviews--list__wrapper")
    if not reviews_wrap:
        print(f"ℹ️ Không thấy reviews: {company_url}")
        return pd.DataFrame()

    rows = []
    elements = reviews_wrap.find_all("article", class_="profile-review")
    print(f"🔍 {company_url} → {len(elements)} reviews")

    for e in elements:
        project_data = extract_review_data(e)
        desc_el = e.find("div", class_="profile-review__summary mobile_hide")
        reviewer_data = extract_reviewer_info(e)

        row: dict = {}
        row.update(project_data or {})
        row.update(reviewer_data or {})
        row["Project description"] = desc_el.get_text(strip=True) if desc_el else None
        row.update(link_social or {})
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    preferred_cols = [
        "reviewer_name",
        "reviewer_role",
        "reviewer_company",
        "verified_status",
        "Industry",
        "Location",
        "Client size",
        "Review type",
        "Services",
        "Project size",
        "Project length",
        "Project description",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [
        c for c in df.columns if c not in preferred_cols
    ]
    return df.loc[:, cols]


# =========================
# Main
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clutch.co crawler")
    p.add_argument(
        "--start-url",
        type=str,
        default=DEFAULT_START_URL,
        help=f"Directory URL bắt đầu (default: {DEFAULT_START_URL})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Số company tối đa để crawl (default: 20)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="clutch_reviews.csv",
        help="Đường dẫn file CSV xuất ra (default: clutch_reviews.csv)",
    )
    p.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint.json',
        help='File checkpoint để lưu trạng thái crawl (default: checkpoint.json)',
    )
    p.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Số worker threads để crawl đồng thời (default: 8)',
    )
    p.add_argument(
        '--flush-every',
        type=int,
        default=20,
        help='Flush kết quả ra file sau mỗi N company (default: 20)',
    )
    p.add_argument(
        '--last_page',
        type=int,
        default=1,
        help='Trang cuối cùng crawl (default: 1)',
    )
    return p.parse_args()   

SLEEP_MIN, SLEEP_MAX = 0.5, 1.5  

# ==== Helper: atomic write CSV ====
def atomic_write_csv(df: pd.DataFrame, path: str, mode: str = "w", header: bool = True) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if mode == "a" and path.exists():
        df.to_csv(tmp, index=False, header=False)
        with open(path, "ab") as fout, open(tmp, "rb") as fin:
            shutil.copyfileobj(fin, fout)
        tmp.unlink(missing_ok=True)
    else:
        df.to_csv(tmp, index=False, header=header)
        tmp.replace(path)

# ==== Checkpoint ====
def load_checkpoint(ckpt_file: str) -> Dict[str, Any]:
    p = Path(ckpt_file)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"done_urls": [], "last_page": 1}
    return {"done_urls": [], "last_page": 1}

def save_checkpoint(ckpt_file: str, state: Dict[str, Any]) -> None:
    p = Path(ckpt_file)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

# ==== Retry wrapper cho get_detail_information ====
def get_detail_information_with_retry(url: str, max_retry: int = 3, backoff_base: float = 0.8) -> pd.DataFrame:
    for attempt in range(1, max_retry + 1):
        try:
            df = get_detail_information(url)  # <- dùng hàm gốc của bạn
            if df is None:
                df = pd.DataFrame()
            return df
        except Exception as e:
            if attempt == max_retry:
                return pd.DataFrame()
            sleep_s = (backoff_base ** attempt) + random.uniform(0.05, 0.25)
            time.sleep(sleep_s)
    return pd.DataFrame()

def process_company(url: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Trả về (train_df, test_df, url) cho 1 company.
    """
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for i in range(2):
        if i == 0:
            url_review = url
        else:
            url_review = f"{url}?page={i}#reviews"

        df = get_detail_information_with_retry(url_review)
        if df is not None and not df.empty and len(df) > 5:
            tr, te = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)
            train_parts.append(tr)
            test_parts.append(te)

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame()
    return train_df, test_df, url

def main() -> None:
    args = parse_args()
    out_path = args.out
    test_out_path = f"{out_path.replace('.csv', '_test.csv')}"
    ckpt_file = getattr(args, "checkpoint", "checkpoint.json")
    workers = max(1, getattr(args, "workers", 8))
    flush_every = max(1, getattr(args, "flush_every", 20)) 

    state = load_checkpoint(ckpt_file)
    done_urls = set(state.get("done_urls", []))
    last_page = int(state.get("last_page", 1))

    if not Path(out_path).exists():
        pd.DataFrame().to_csv(out_path, index=False)
    if not Path(test_out_path).exists():
        pd.DataFrame().to_csv(test_out_path, index=False)

    for com in range(last_page, 2):
        start_url = args.start_url
        # (Gợi ý) Có thể muốn set page luôn (không chỉ com==1) tùy site:
        if com > 1:
            start_url = f"{start_url}?page={com}"

        company_urls = get_base_url_company(start_url)
        if not company_urls:
            print("⚠️ Không có URL công ty nào được tìm thấy. Kết thúc.", file=sys.stderr)
            state["last_page"] = com + 1
            save_checkpoint(ckpt_file, state)
            return

        company_urls = [u for u in company_urls if u not in done_urls]
        print("=========================>")
        print(f"🔎 Trang {com}: còn {len(company_urls)} công ty cần crawl (tổng trang có thể lớn hơn).")

        batch_trains: List[pd.DataFrame] = []
        batch_tests: List[pd.DataFrame] = []
        futures = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for url in company_urls:
                futures.append(ex.submit(process_company, url))

            # Thu kết quả theo từng company, flush mỗi 'flush_every'
            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Page {com}")
            for fut in pbar:
                tr_df, te_df, url = fut.result()
                if tr_df is not None and not tr_df.empty:
                    batch_trains.append(tr_df)
                if te_df is not None and not te_df.empty:
                    batch_tests.append(te_df)
                done_urls.add(url)
                # print('URL done:', url)
                # print('TR_df :',len(tr_df), 'TE_df :', len(te_df))
                # print('Batch trains:', len(batch_trains), 'Batch tests:', len(batch_tests))
                # print('Flush every:', flush_every)
                if len(batch_trains) >= flush_every or len(batch_tests) >= flush_every:
                    if batch_trains:
                        big_tr = pd.concat(batch_trains, ignore_index=True)
                        atomic_write_csv(big_tr, out_path, mode="a", header=False)
                        batch_trains.clear()
                    if batch_tests:
                        big_te = pd.concat(batch_tests, ignore_index=True)
                        atomic_write_csv(big_te, test_out_path, mode="a", header=False)
                        batch_tests.clear()

                    state["done_urls"] = list(done_urls)
                    state["last_page"] = com
                    save_checkpoint(ckpt_file, state)

        if batch_trains:
            big_tr = pd.concat(batch_trains, ignore_index=True)
            atomic_write_csv(big_tr, out_path, mode="a", header=False)
        if batch_tests:
            big_te = pd.concat(batch_tests, ignore_index=True)
            atomic_write_csv(big_te, test_out_path, mode="a", header=False)

        state["done_urls"] = list(done_urls)
        state["last_page"] = com + 1
        save_checkpoint(ckpt_file, state)

    print(f"✅ Hoàn tất. Kết quả ở: {out_path} và {test_out_path}")
    
if __name__ == "__main__":
    main()