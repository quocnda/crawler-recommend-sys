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

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    # Fallback khi không cài tqdm
    def tqdm(x, **kwargs):
        return x

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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lấy danh sách company URLs
    company_urls = get_base_url_company(args.start_url)
    if not company_urls:
        print("⚠️ Không có URL công ty nào được tìm thấy. Kết thúc.", file=sys.stderr)
        return

    # limit = max(0, args.limit or 0)
    # if limit:
    #     company_urls = company_urls[:limit]

    all_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for url in tqdm(company_urls, desc="Crawling companies"):
        df = get_detail_information(url)
        if not df.empty:
            train_data, test_data = train_test_split(df, test_size=0.3)
            all_df = pd.concat([all_df, train_data], ignore_index=True)
            test_df = pd.concat([test_df,test_data], ignore_index=True)
        # ngủ ngẫu nhiên giảm rate-limit
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    if not all_df.empty:
        all_df.to_csv(args.out, index=False)
        test_df.to_csv(f"{(args.out).replace('.csv', '_test.csv')}", index=False)
        print(f"✅ Đã ghi {len(all_df)} dòng vào: {args.out}")
    else:
        print("⚠️ Không thu được dữ liệu (có thể bị anti-bot chặn).", file=sys.stderr)


if __name__ == "__main__":
    main()
