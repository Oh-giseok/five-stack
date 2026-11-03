"""Inspect raw availability and integrity without producing model inputs."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from oinq.collectors.raw import month_ranges, months_in_range, utc_now, write_json


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_raw(root, config, snapshot):
    root = Path(root)
    folder = root / config["sources"]["storage"]["raw_dir"] / snapshot
    reports = root / config["sources"]["storage"]["reports_dir"]
    country_rows = config["countries"]["countries"]
    countries = {row["iso3"]: row for row in country_rows}
    customs_country = {row["customs"]: row["iso3"] for row in country_rows}
    reporter_country = {row["comtrade_reporter"]: row["iso3"] for row in country_rows}
    partner_country = {row["comtrade_partner"]: row["iso3"] for row in country_rows}
    products = {row["hs_code"] for row in config["products"]["products"]}
    period = config["splits"]["collection"]
    months = set(months_in_range(period["monthly_start"], period["monthly_end"]))
    years = set(range(int(period["monthly_start"][:4]), int(period["monthly_end"][:4]) + 1))
    present = defaultdict(set)
    amount_present, weight_present, zero_amount = defaultdict(set), defaultdict(set), defaultdict(set)
    counts = Counter()
    custom_keys, comtrade_keys = set(), set()
    duplicate_keys, integrity_errors, api_errors, total_differences = [], [], [], []
    raw_files = []
    estimated_counts = Counter()

    for meta_path in sorted(folder.glob("*/*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.stem.startswith("probe_"):
            continue
        if not meta.get("transport_ok") or not meta.get("api_ok", True):
            api_errors.append(str(meta_path.relative_to(folder)))
            continue
        body_path = meta_path.parent / meta["body_file"]
        if not body_path.is_file() or hashlib.sha256(body_path.read_bytes()).hexdigest() != meta["sha256"]:
            integrity_errors.append(str(body_path.relative_to(folder)))
            continue
        raw_files.append({"path": str(body_path.relative_to(folder)), "sha256": meta["sha256"],
                          "bytes": meta["bytes"], "collected_at": meta["collected_at"]})
        if meta["source"] == "customs":
            document = ET.fromstring(body_path.read_bytes())
            if document.findtext(".//resultCode") != "00":
                api_errors.append(str(body_path.relative_to(folder)))
                continue
            rows = [{child.tag: child.text for child in item} for item in document.findall(".//item")]
            details = [row for row in rows if row.get("year") != "총계"]
            for row in details:
                country = customs_country[row["statCd"]]
                code, month = row["hsCd"][:6], row["year"].replace(".", "")
                if code not in products or month not in months:
                    api_errors.append(str(body_path.relative_to(folder)))
                    continue
                key = ("customs_export", country, code)
                present[key].add(month)
                if row.get("expDlr") is not None:
                    amount_present[key].add(month)
                    if int(row["expDlr"]) == 0:
                        zero_amount[key].add(month)
                if row.get("expWgt") is not None:
                    weight_present[key].add(month)
                unique_key = (country, row["hsCd"], month)
                if unique_key in custom_keys:
                    duplicate_keys.append(["customs", *unique_key])
                custom_keys.add(unique_key)
                counts["customs_detail_rows"] += 1
            for total in (row for row in rows if row.get("year") == "총계"):
                counts["customs_total_rows"] += 1
                for field in ("expDlr", "expWgt", "impDlr", "impWgt"):
                    difference = int(total[field]) - sum(int(row[field]) for row in details)
                    if difference:
                        total_differences.append({"file": body_path.name, "field": field, "difference": difference})
            counts["customs_requests"] += 1
        elif meta["source"] == "comtrade":
            document = json.loads(body_path.read_bytes())
            rows = document.get("data", [])
            if document.get("error") or len(rows) >= 100000 or document.get("count") != len(rows):
                api_errors.append(str(body_path.relative_to(folder)))
            for row in rows:
                if row["flowCode"] == "X" and row["reporterCode"] == 410:
                    country, kind = partner_country[row["partnerCode"]], "comtrade_export_comparison"
                else:
                    country = reporter_country[row["reporterCode"]]
                    kind = "comtrade_world_import" if row["partnerCode"] == 0 else "comtrade_korea_import"
                key = (kind, country, str(row["cmdCode"]))
                month = str(row["period"])
                present[key].add(month)
                if row.get("primaryValue") is not None:
                    amount_present[key].add(month)
                    if row["primaryValue"] == 0:
                        zero_amount[key].add(month)
                if row.get("netWgt") is not None:
                    weight_present[key].add(month)
                unique_key = (kind, country, row["cmdCode"], month)
                if unique_key in comtrade_keys:
                    duplicate_keys.append(list(unique_key))
                comtrade_keys.add(unique_key)
                counts[kind + "_rows"] += 1
                if row.get("isNetWgtEstimated"):
                    estimated_counts[kind + "_estimated_weight_rows"] += 1
            counts["comtrade_requests"] += 1
        elif meta["source"] == "world_bank":
            counts["world_bank_pages"] += 1

    coverage = []
    for kind in ("customs_export", "comtrade_export_comparison", "comtrade_world_import", "comtrade_korea_import"):
        for country in countries:
            for product in sorted(products):
                key = (kind, country, product)
                available = present[key]
                coverage.append({"source_role": kind, "country_code": country, "hs_code": product,
                                 "expected_months": len(months), "response_months": len(available),
                                 "amount_present_months": len(amount_present[key]),
                                 "weight_present_months": len(weight_present[key]),
                                 "months_with_an_explicit_zero_detail": len(zero_amount[key]),
                                 "first_month": min(available, default=""), "last_month": max(available, default=""),
                                 "observed_2026_months": len({month for month in available if month.startswith("2026")}),
                                 "missing_months": ";".join(sorted(months - available))})
    macro_summary = json.loads((folder / "world_bank_summary.json").read_text(encoding="utf-8"))
    macro_coverage = []
    for indicator, info in macro_summary["indicators"].items():
        for country, detail in info["countries"].items():
            macro_coverage.append({"indicator": indicator, "country_code": country, "expected_years": len(years),
                                   **{key: value for key, value in detail.items() if key != "missing_years"},
                                   "missing_years": ";".join(map(str, detail["missing_years"]))})
    role_summaries = {}
    for kind in sorted({row["source_role"] for row in coverage}):
        subset = [row for row in coverage if row["source_role"] == kind]
        role_summaries[kind] = {"country_product_combinations": len(subset),
                                "observed_month_cells": sum(row["response_months"] for row in subset),
                                "expected_month_cells": len(subset) * len(months),
                                "complete_combinations": sum(row["response_months"] == len(months) for row in subset),
                                "complete_2026_q1_q3_combinations": sum(row["observed_2026_months"] == 9 for row in subset)}
    expected_requests = {"customs_requests": len(countries) * len(products) * len(month_ranges(config)),
                         "comtrade_requests": (len(countries) + 1) * len(month_ranges(config))}
    collection_complete = (all(counts[key] == expected for key, expected in expected_requests.items())
                           and macro_summary.get("status") == "downloaded_with_missing_values"
                           and set(macro_summary["indicators"]) == set(config["sources"]["world_bank"]["indicators"]))
    summary = {"snapshot": snapshot, "audited_at": utc_now(), "requested_start": period["monthly_start"],
               "requested_end": period["monthly_end"], "preprocessing_applied": False,
               "counts": dict(counts), "expected_requests": expected_requests, "collection_complete": collection_complete,
               "roles": role_summaries, "estimated_flags": dict(estimated_counts),
               "integrity_errors": integrity_errors, "api_errors": api_errors,
               "duplicate_keys": duplicate_keys, "source_total_differences": total_differences,
               "macro_non_null_values": sum(row["non_null_values"] for row in macro_coverage),
               "macro_expected_values": len(macro_coverage) * len(years),
               "config": config, "files": raw_files}
    summary["status"] = "audited_with_gaps" if not integrity_errors and not api_errors and not duplicate_keys else "review_required"
    if not collection_complete:
        summary["status"] = "collection_incomplete"
    write_json(folder / "collection_manifest.json", summary)
    write_csv(reports / f"{snapshot}_trade_coverage.csv", coverage)
    write_csv(reports / f"{snapshot}_macro_coverage.csv", macro_coverage)
    if collection_complete:
        write_collection_report(reports / f"{snapshot}.md", summary, coverage, macro_coverage, countries)
    print(json.dumps({key: summary[key] for key in ("status", "counts", "roles", "macro_non_null_values", "macro_expected_values")}, ensure_ascii=False, indent=2))
    return summary


def write_collection_report(path, summary, coverage, macro_coverage, countries):
    """Publish observed coverage only; all monetary targets stay in raw files."""
    counts, roles = summary["counts"], summary["roles"]
    lines = ["# 실데이터 원본 수집 결과", "",
             f"수집 스냅샷: `{summary['snapshot']}` | 점검시각(UTC): {summary['audited_at']}", "",
             f"요청 범위: {summary['requested_start']}~{summary['requested_end']}, 20개국·8개 HS6. "
             "원본 응답 수집과 확보 범위 점검까지만 수행했다. 분기 집계·결측 대체·단가·점유율·시차·학습 파일은 생성하지 않았다.", "",
             "## 수집 결과", "",
             "| 원천·용도 | 완료 요청 | 원본 상세 레코드 | 월별 국가×제품 행 확보 |",
             "|---|---:|---:|---:|"]
    specifications = [
        ("관세청 한국 수출", "customs_requests", "customs_detail_rows", "customs_export"),
        ("Comtrade 한국 수출 비교", None, "comtrade_export_comparison_rows", "comtrade_export_comparison"),
        ("Comtrade 대상국 전체 수입", None, "comtrade_world_import_rows", "comtrade_world_import"),
        ("Comtrade 대상국 한국산 수입", None, "comtrade_korea_import_rows", "comtrade_korea_import"),
    ]
    for label, request_key, record_key, role in specifications:
        info = roles[role]
        request_count = str(counts.get(request_key, 0)) if request_key else "아래 합계 참고"
        lines.append(f"| {label} | {request_count} | {counts.get(record_key, 0):,} | "
                     f"{info['observed_month_cells']:,}/{info['expected_month_cells']:,} |")
    lines.extend(["", f"Comtrade 전체 완료 요청은 {counts.get('comtrade_requests', 0)}개다. "
                  f"관세청 합계 행 {counts.get('customs_total_rows', 0):,}개는 원본에 보존했으며 상세 레코드 수에 더하지 않았다.", "",
                  "요청 완료는 모든 월의 거래 행이 존재한다는 뜻이 아니다. 미반환 월은 미거래·미신고·미공개를 추가로 구분해야 하며 현재 0으로 채우지 않았다.", "",
                  "## 국가별 확보 범위", "",
                  "| 국가 | 한국 수출 월행/1,320 | 수출 2026Q1~Q3 월행/72 | 전체 수입 월행/1,320 | 한국산 수입 월행/1,320 | 전체 수입 최신 월 |",
                  "|---|---:|---:|---:|---:|---|"])
    for code, country in countries.items():
        subset = [row for row in coverage if row["country_code"] == code]
        by_role = {role: [row for row in subset if row["source_role"] == role] for role in roles}
        observed = lambda role: sum(row["response_months"] for row in by_role[role])
        latest_import = max((row["last_month"] for row in by_role["comtrade_world_import"]), default="")
        current_export = sum(row["observed_2026_months"] for row in by_role["customs_export"])
        lines.append(f"| {country['name_ko']} ({code}) | {observed('customs_export'):,} | {current_export} | "
                     f"{observed('comtrade_world_import'):,} | {observed('comtrade_korea_import'):,} | {latest_import or '없음'} |")
    lines.extend(["", "월행은 해당 국가×HS6×월에 상세 응답이 하나 이상 있는지를 센 값이다. "
                  "HS10 상세 행 수나 수출금액 합계와는 다르며 학습 행 수가 아니다.", "",
                  f"관세청에서 2026년 1~9월의 모든 월에 응답 행이 있는 조합은 "
                  f"{roles['customs_export']['complete_2026_q1_q3_combinations']}/160개다. "
                  "나머지는 위 CSV의 누락 월을 확인한 뒤 테스트 포함 여부를 판단한다.", "",
                  "## 거시지표", "",
                  "| 독립변수 | 값 존재/요청 연도·국가 수 | 최신 관측 연도 |",
                  "|---|---:|---|"])
    for name in summary["config"]["sources"]["world_bank"]["indicators"]:
        subset = [row for row in macro_coverage if row["indicator"] == name]
        latest = sorted({row["last_year"] for row in subset if row["last_year"] is not None})
        lines.append(f"| {name} | {sum(row['non_null_values'] for row in subset)}/{sum(row['expected_years'] for row in subset)} | "
                     f"{', '.join(map(str, latest))} |")
    lines.extend(["", "2026년 연간 값은 조회 결과에 없었다. 인터넷 이용률 등 일부 지표는 2025년 값도 없다. "
                  "국가·지표별 누락 연도는 macro coverage CSV에 기록했다. World Bank 현재 수정치의 과거 공개시점을 증명한 상태는 아니다.", "",
                  "## 타깃·독립변수와 원본의 연결", "",
                  "| 항목 | 확보한 원본 | 현재 상태 |", "|---|---|---|",
                  "| 타깃 수출금액 | 관세청 expDlr | 월별 HS10 원본 확보; HS6·분기 합산은 후속 |",
                  "| 수출중량 | 관세청 expWgt | kg 원본 확보 |",
                  "| 수출단가 | 수출금액·중량 원본 | 아직 계산하지 않음 |",
                  "| 시장 규모·한국산 수입 | Comtrade primaryValue, reporter·partner·flow | 원본 확보 범위는 위 표·CSV 참고 |",
                  "| 한국산 점유율 | 같은 대상국의 전체·한국산 수입 | 아직 계산하지 않음 |",
                  "| 거시지표·K-means 지표 | World Bank 연간 6종 | 원래 값·null·연도 보존; 군집용 표준화 미수행 |",
                  "| 식별·시간 | YAML 국가·HS, 원본 월·연도 | 공통 학습 테이블 미생성 |",
                  "| 시차·이동 통계·성장률 | 위 원본 교역 시계열 | 아직 생성하지 않음 |", "",
                  "## 무결성·후속 확인", "",
                  f"- 체크섬 점검 원본 {len(summary['files']):,}개. 무결성 오류 {len(summary['integrity_errors'])}개, "
                  f"API 오류 파일 {len(summary['api_errors'])}개, 중복 키 {len(summary['duplicate_keys'])}개."])
    differences = summary["source_total_differences"]
    for field in ("expDlr", "expWgt", "impDlr", "impWgt"):
        subset = [row for row in differences if row["field"] == field]
        lines.append(f"- 관세청 {field}: 합계·상세 차이 {len(subset)}건, 절대 차이 최댓값 "
                     f"{max((abs(row['difference']) for row in subset), default=0)}. 원본을 수정하지 않았다.")
    lines.extend(["- 중량 차이는 원천 반올림 등 여부를 후속 확인한다. 관세청 응답에 HS 판본명이 없어 판본 확인도 전처리 전 점검 대상이다.",
                  "- Comtrade 중량 추정 플래그와 CIF/FOB 필드는 원본에 보존했다. 수출 비교 자료를 관세청 금액에 중복 합산하지 않았다.",
                  "- 인증키·원본·최종 테스트 정답은 Git 공유 대상에서 제외했다. 공통 학습 데이터 manifest는 계속 not_generated 상태다.", "",
                  "## 파일 위치", "",
                  f"- 원본·요청 메타정보·collection_manifest.json: `data/raw/{summary['snapshot']}/`",
                  f"- 교역 확보 범위: [{summary['snapshot']}_trade_coverage.csv]({summary['snapshot']}_trade_coverage.csv)",
                  f"- 거시 확보 범위: [{summary['snapshot']}_macro_coverage.csv]({summary['snapshot']}_macro_coverage.csv)", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
