"""Save unmodified API payloads and request evidence; no feature engineering."""

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path
import time
import threading
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from dotenv import dotenv_values
import requests


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class AccessError(RuntimeError):
    """A request cannot proceed without user help or a quota reset."""


class RawClient:
    def __init__(self, root, config, snapshot):
        self.root = Path(root)
        self.config = config
        self.folder = self.root / config["sources"]["storage"]["raw_dir"] / snapshot
        self.folder.mkdir(parents=True, exist_ok=True)
        self.env = dotenv_values(self.root / ".env")
        self.thread_local = threading.local()
        self.rate_lock = threading.Lock()
        self.last_request = {}
        self.pause_until = {}
        self.requests_made = 0

    def session(self):
        if not hasattr(self.thread_local, "session"):
            session = requests.Session()
            session.headers["User-Agent"] = "oinq-education-data-collection/0.1"
            self.thread_local.session = session
        return self.thread_local.session

    def key(self, source):
        name = self.config["sources"][source]["api_key_env"]
        value = os.environ.get(name) or self.env.get(name)
        if not value or not value.strip():
            raise AccessError(f"{name}: 로컬 인증키가 필요합니다.")
        return value.strip()

    def fetch(self, source, name, endpoint, params, *, secret_params=None, headers=None, kind="json"):
        """Resume successful bodies; failures are recorded without secret URLs."""
        folder = self.folder / source
        folder.mkdir(exist_ok=True)
        metadata_path = folder / f"{name}.meta.json"
        body_path = folder / f"{name}.{kind}"
        if metadata_path.exists() and body_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (previous.get("endpoint") != endpoint or previous.get("parameters") != params):
                raise AccessError(f"{source}: 기존 스냅샷과 요청 조건이 다릅니다. 새 snapshot이 필요합니다.")
            if (previous.get("transport_ok") and previous.get("api_ok", True)
                    and previous.get("sha256") == hashlib.sha256(body_path.read_bytes()).hexdigest()):
                return body_path.read_bytes(), previous
        settings = self.config["sources"]["request_defaults"]
        secrets = list((secret_params or {}).values()) + list((headers or {}).values())
        request_params = {**params, **(secret_params or {})}
        metadata = {"source": source, "endpoint": endpoint, "parameters": params,
                    "authentication_configured": bool(secrets), "collected_at": None,
                    "transport_ok": False, "preprocessing_applied": False}
        for attempt in range(settings["max_retries"] + 1):
            interval = self.config["sources"][source].get("minimum_interval_seconds", settings["minimum_interval_seconds"])
            with self.rate_lock:
                elapsed = time.monotonic() - self.last_request.get(source, 0)
                time.sleep(max(0, interval - elapsed, self.pause_until.get(source, 0) - time.monotonic()))
                self.last_request[source] = time.monotonic()
                self.requests_made += 1
            try:
                response = self.session().get(endpoint, params=request_params, headers=headers,
                                            timeout=settings["timeout_seconds"])
            except requests.RequestException as exc:
                metadata.update(collected_at=utc_now(), error_type=type(exc).__name__)
                write_json(metadata_path, metadata)
                if attempt == settings["max_retries"]:
                    raise AccessError(f"{source}: 네트워크 요청 실패 ({type(exc).__name__})") from None
            else:
                body = response.content
                redacted = False
                for secret in secrets:
                    if secret and str(secret).encode() in body:
                        body = body.replace(str(secret).encode(), b"[REDACTED]")
                        redacted = True
                body_path.write_bytes(body)
                metadata.update(collected_at=utc_now(), http_status=response.status_code,
                                transport_ok=response.ok, sha256=hashlib.sha256(body).hexdigest(),
                                body_file=body_path.name, bytes=len(body), secret_echo_redacted=redacted,
                                server_date=response.headers.get("Date"),
                                retry_after=response.headers.get("Retry-After"),
                                content_type=response.headers.get("Content-Type"), attempt=attempt + 1)
                write_json(metadata_path, metadata)
                if response.status_code in (401, 403):
                    raise AccessError(f"{source}: HTTP {response.status_code}; 인증·권한·할당량 확인이 필요합니다.")
                if response.status_code == 429:
                    match = re.search(r"[Tt]ry again in (\d+) seconds?", body.decode("utf-8", errors="replace"))
                    retry_text = response.headers.get("Retry-After", "")
                    retry_after = int(retry_text) if retry_text.isdigit() else int(match.group(1)) if match else None
                    if retry_after is None or retry_after > 60 or attempt == settings["max_retries"]:
                        raise AccessError(f"{source}: HTTP 429; 할당량·대기시간 확인이 필요합니다 (retry_after={retry_after}).")
                    wait_seconds = max(5, retry_after + 2)
                    with self.rate_lock:
                        self.pause_until[source] = time.monotonic() + wait_seconds
                    print(f"{source}: 단기 요청 제한, {wait_seconds}초 대기 후 제한된 횟수로 재시도", flush=True)
                    continue
                if response.ok:
                    return body, metadata
                if response.status_code not in settings["retry_status_codes"]:
                    raise AccessError(f"{source}: HTTP {response.status_code}; 요청 조건 확인이 필요합니다.")
                if attempt == settings["max_retries"]:
                    raise AccessError(f"{source}: HTTP {response.status_code}; 재시도 후에도 실패했습니다.")
            time.sleep(settings["retry_backoff_seconds"][min(attempt, len(settings["retry_backoff_seconds"]) - 1)])

    def mark_api(self, source, name, metadata, ok):
        metadata["api_ok"] = ok
        write_json(self.folder / source / f"{name}.meta.json", metadata)


def customs_request(client, country, product, start, end, name):
    settings = client.config["sources"]["customs"]
    fields = settings["request_fields"]
    params = {fields["start"]: start.replace("-", ""), fields["end"]: end.replace("-", ""),
              fields["country"]: country[settings["country_mapping_field"]], fields["hs_code"]: product["hs_code"]}
    body, meta = client.fetch("customs", name, settings["endpoint"], params,
                              secret_params={settings["api_key_parameter"]: unquote(client.key("customs"))}, kind="xml")
    document = ET.fromstring(body)
    code = document.findtext(".//resultCode")
    client.mark_api("customs", name, meta, code == "00")
    if code != "00":
        raise AccessError(f"customs: API 결과코드 {code or document.findtext('.//returnReasonCode') or '없음'}; 수집을 중지합니다.")
    rows = [{child.tag: child.text for child in item} for item in document.findall(".//item")]
    return rows, meta


def comtrade_request(client, reporter, partners, products, periods, flow, name):
    settings = client.config["sources"]["comtrade"]
    params = {"reporterCode": str(reporter), "partnerCode": ",".join(map(str, partners)),
              "cmdCode": ",".join(products), "period": ",".join(periods), "flowCode": flow,
              "partner2Code": settings["partner2_code"], "customsCode": settings["customs_code"],
              "motCode": settings["mode_of_transport_code"], "maxRecords": 100000,
              "includeDesc": "true"}
    body, meta = client.fetch("comtrade", name, settings["data_endpoint"], params,
                              headers={settings["api_key_header"]: client.key("comtrade")})
    document = json.loads(body)
    rows = document.get("data")
    if not isinstance(rows, list):
        client.mark_api("comtrade", name, meta, False)
        raise AccessError("comtrade: data 배열이 없는 응답입니다.")
    if document.get("error") or len(rows) >= params["maxRecords"]:
        client.mark_api("comtrade", name, meta, False)
        raise AccessError("comtrade: API 오류 또는 응답 제한 의심; 수집을 중지합니다.")
    client.mark_api("comtrade", name, meta, True)
    return rows, meta


def probe(client):
    countries = client.config["countries"]["countries"]
    country = next(row for row in countries if row["iso3"] == "USA")
    product = next(row for row in client.config["products"]["products"] if row["hs_code"] == "330499")
    results = {}
    try:
        rows, _ = customs_request(client, country, product, "2025-01", "2025-01", "probe_USA_330499_202501")
        results["customs"] = {"status": "success", "records": len(rows), "sample": rows[:2]}
    except (AccessError, ET.ParseError, ValueError) as exc:
        results["customs"] = {"status": "failed", "message": str(exc)}
    try:
        rows, _ = comtrade_request(client, country["comtrade_reporter"], [0, 410], [product["hs_code"]],
                                   ["202501"], "M", "probe_USA_330499_202501")
        results["comtrade"] = {"status": "success", "records": len(rows), "sample": rows[:2]}
    except (AccessError, ValueError) as exc:
        results["comtrade"] = {"status": "failed", "message": str(exc)}
    settings = client.config["sources"]["world_bank"]
    try:
        body, _ = client.fetch("world_bank", "probe_USA_gdp_2024",
                               settings["base_url"] + "/country/USA/indicator/NY.GDP.MKTP.CD",
                               {"format": "json", "source": settings["source_id"], "date": "2024", "per_page": 1000})
        document = json.loads(body)
        results["world_bank"] = {"status": "success", "response": document}
    except (AccessError, ValueError) as exc:
        results["world_bank"] = {"status": "failed", "message": str(exc)}
    write_json(client.folder / "probe_summary.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


def collect_world_bank(client):
    """Download all requested country/indicator/year rows, retaining nulls."""
    settings = client.config["sources"]["world_bank"]
    countries = client.config["countries"]["countries"]
    start = client.config["splits"]["collection"]["monthly_start"][:4]
    end = client.config["splits"]["collection"]["monthly_end"][:4]
    country_codes = ";".join(row["world_bank"] for row in countries)
    summary = {"source": "world_bank", "status": "collecting", "indicators": {},
               "requested_years": [int(start), int(end)], "preprocessing_applied": False}
    for column, indicator in settings["indicators"].items():
        page, pages, rows = 1, 1, []
        while page <= pages:
            endpoint = settings["base_url"] + settings["endpoint_template"].format(
                country=country_codes, indicator=indicator["code"])
            body, _ = client.fetch("world_bank", f"{column}_{start}_{end}_page{page}", endpoint,
                                   {"date": f"{start}:{end}", "format": "json", "source": settings["source_id"],
                                    "per_page": settings["page_size"], "page": page})
            document = json.loads(body)
            if not isinstance(document, list) or len(document) != 2 or not isinstance(document[1], list):
                raise AccessError(f"world_bank: {column} 응답 형식 오류")
            pages = int(document[0]["pages"])
            rows.extend(document[1])
            page += 1
        by_country = {}
        for country in countries:
            subset = [row for row in rows if row["countryiso3code"] == country["iso3"]]
            observed = [int(row["date"]) for row in subset if row["value"] is not None]
            by_country[country["iso3"]] = {
                "response_rows": len(subset), "non_null_values": len(observed),
                "first_year": min(observed, default=None), "last_year": max(observed, default=None),
                "missing_years": sorted(set(range(int(start), int(end) + 1)) - set(observed))}
        summary["indicators"][column] = {"code": indicator["code"], "response_rows": len(rows),
                                         "non_null_values": sum(row["value"] is not None for row in rows),
                                         "countries": by_country}
        write_json(client.folder / "world_bank_summary.json", summary)
        print(f"world_bank {column}: rows={len(rows)}, values={summary['indicators'][column]['non_null_values']}", flush=True)
    summary.update(status="downloaded_with_missing_values", completed_at=utc_now())
    write_json(client.folder / "world_bank_summary.json", summary)
    return summary


def month_ranges(config):
    start = config["splits"]["collection"]["monthly_start"]
    end = config["splits"]["collection"]["monthly_end"]
    return [(max(start, f"{year}-01"), min(end, f"{year}-12"))
            for year in range(int(start[:4]), int(end[:4]) + 1)]


def months_in_range(start, end):
    months = []
    year, month = map(int, start.split("-"))
    while f"{year:04d}-{month:02d}" <= end:
        months.append(f"{year:04d}{month:02d}")
        month += 1
        if month == 13:
            month, year = 1, year + 1
    return months


def collect_customs(client):
    """Request each exact HS6/country in <=12-month windows; keep HS10 raw."""
    jobs = [(country, product, start, end)
            for country in client.config["countries"]["countries"]
            for product in client.config["products"]["products"]
            for start, end in month_ranges(client.config)]
    summary = {"source": "customs", "status": "collecting", "planned_requests": len(jobs),
               "completed_requests": 0, "detail_rows": 0, "preprocessing_applied": False,
               "failures": [], "total_mismatches": [], "started_at": utc_now()}
    stop = threading.Event()

    def run(job):
        if stop.is_set():
            return None
        country, product, start, end = job
        name = f"{country['iso3']}_{product['hs_code']}_{start.replace('-', '')}_{end.replace('-', '')}"
        rows, _ = customs_request(client, country, product, start, end, name)
        details = [row for row in rows if row.get("year") != "총계"]
        allowed_months = set(months_in_range(start, end))
        for row in details:
            if (row.get("statCd") != country["customs"]
                    or not str(row.get("hsCd", "")).startswith(product["hs_code"])
                    or row.get("year", "").replace(".", "") not in allowed_months):
                raise AccessError(f"customs: {name} 응답의 국가·품목·월이 요청과 다릅니다.")
        totals = [row for row in rows if row.get("year") == "총계"]
        mismatch = []
        for total in totals:
            for field in ("expDlr", "expWgt", "impDlr", "impWgt"):
                if sum(int(row[field]) for row in details) != int(total[field]):
                    mismatch.append(field)
        return name, len(details), mismatch

    workers = client.config["sources"]["customs"].get("max_concurrent_requests", 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        try:
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                name, count, mismatch = result
                summary["completed_requests"] += 1
                summary["detail_rows"] += count
                if mismatch:
                    summary["total_mismatches"].append({"request": name, "fields": mismatch})
                if summary["completed_requests"] % 50 == 0:
                    write_json(client.folder / "customs_summary.json", summary)
                    print(f"customs {summary['completed_requests']}/{len(jobs)} requests; detail_rows={summary['detail_rows']}", flush=True)
        except Exception as exc:
            stop.set()
            for future in futures:
                future.cancel()
            summary["failures"].append({"type": type(exc).__name__, "message": str(exc)})
            summary["status"] = "interrupted"
            write_json(client.folder / "customs_summary.json", summary)
            raise
    summary.update(status="downloaded", completed_at=utc_now())
    write_json(client.folder / "customs_summary.json", summary)
    print(f"customs finished: {summary['completed_requests']} requests, {summary['detail_rows']} detail rows", flush=True)
    return summary


def collect_comtrade(client):
    """Stay below the registered free quota with 12-month/8-product batches."""
    countries = client.config["countries"]["countries"]
    products = [row["hs_code"] for row in client.config["products"]["products"]]
    settings = client.config["sources"]["comtrade"]
    jobs = []
    # Primary export series is still customs; this is a separately retained comparison.
    for start, end in month_ranges(client.config):
        periods = months_in_range(start, end)
        jobs.append(("KOR_exports", 410, [row["comtrade_partner"] for row in countries], periods, "X"))
        for country in countries:
            jobs.append((f"{country['iso3']}_imports", country["comtrade_reporter"], [0, 410], periods, "M"))
    summary = {"source": "comtrade", "status": "collecting", "planned_requests": len(jobs),
               "completed_requests": 0, "records": 0, "empty_requests": [],
               "preprocessing_applied": False, "started_at": utc_now()}
    stop = threading.Event()

    def run(job):
        if stop.is_set():
            return None
        label, reporter, partners, periods, flow = job
        name = f"{label}_{periods[0]}_{periods[-1]}"
        rows, meta = comtrade_request(client, reporter, partners, products, periods, flow, name)
        for row in rows:
            if (row["reporterCode"] != reporter or row["partnerCode"] not in partners
                    or str(row["cmdCode"]) not in products or str(row["period"]) not in periods
                    or row["flowCode"] != flow or row["partner2Code"] != settings["partner2_code"]
                    or row["customsCode"] != settings["customs_code"] or row["motCode"] != settings["mode_of_transport_code"]):
                raise AccessError(f"comtrade: {name} 응답 조건이 요청과 다릅니다.")
        return name, len(rows)

    workers = settings.get("max_concurrent_requests", 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        try:
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                name, count = result
                summary["completed_requests"] += 1
                summary["records"] += count
                if not count:
                    summary["empty_requests"].append(name)
                write_json(client.folder / "comtrade_summary.json", summary)
                print(f"comtrade {summary['completed_requests']}/{len(jobs)}: {name}, rows={count}", flush=True)
        except Exception as exc:
            stop.set()
            for future in futures:
                future.cancel()
            summary.update(status="interrupted", error_type=type(exc).__name__, message=str(exc))
            write_json(client.folder / "comtrade_summary.json", summary)
            raise
    summary.update(status="downloaded", completed_at=utc_now())
    write_json(client.folder / "comtrade_summary.json", summary)
    return summary
