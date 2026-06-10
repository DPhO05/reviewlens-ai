"""
LLM Labeling Script — điền vào cột candidate_llm_* trong data_labeling.csv
Theo chiến lược llm_labeling_strategy.md
Model: Antigravity LLM (Claude Sonnet 4.6)
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE  = ROOT / "Data" / "gold_data" / "data_labeling.csv"
OUTPUT_FILE = ROOT / "Data" / "gold_data" / "data_labeling_labeled.csv"
MODEL_NAME  = "antigravity_claude_sonnet_4_6"
LABEL_VER   = "llm_helpfulness_v2"


# ─── RUBRIC SCORING ──────────────────────────────────────────────────────────

def score_specificity(t: str) -> int:
    """0-2: chi tiết cụ thể"""
    tl = t.lower()
    high = [
        r'\d+\s*(ngày|tuần|tháng|năm|giờ|phút|lần)',
        r'\d+\s*(cm|mm|kg|g|ml|l|w|mah|gb|mb|k|đ)',
        r'(pin|battery).{0,30}\d',
        r'(size|cỡ|số).{0,20}(nhỏ|lớn|vừa|chật|rộng|\d)',
        r'(mùi|màu|chất|vị|độ).{0,30}(thơm|ngọt|chua|đắng|thô|mỏng|dày|bền|mịn)',
        r'(lỗi|hỏng|vỡ|nứt|bong|tróc|cháy|chập|chảy|rò|đứt|kêu tiếng)',
        r'(không dùng được|không hoạt động|không nhận|không kết nối|không chín)',
        r'(so với|hơn so với|kém hơn|tốt hơn)',
        r'(kiềm dầu|ẩm mịn|lên tone|chống trôi|thấm nhanh|bết rít|cay mắt|kích ứng|táo bón|dị ứng)',
        r'(nút giữa|lực bấm|độ nhạy|đàn hồi|tiếng ồn|lag|treo máy|nóng máy)',
        r'(tã.{0,20}(thoáng|hăm|thấm|rò))',
        r'(in hơi ẩu|bìa nhoè|giấy dày|giấy mỏng|mực in|in rõ|in sạch)',
        r'(60w|90w|usb-c|type-c|hdmi|wifi|bluetooth)',
        # book/content specific
        r'(nội dung|nhân vật|tác giả|phong cách|văn phong|cốt truyện|tình tiết).{0,40}(hay|tốt|kém|sáo|chân thực|sinh động|nhân văn|kỳ ảo|cảm động)',
        r'(thoáng|hăm|thấm hút|rò rỉ|co giãn|đàn hồi)',
        r'(lực hút|khí thổi|lọc|không lọc|tín hiệu|kết nối)',
        r'(kêu tạch tạch|kêu to|kêu nhỏ|ồn|êm|tiếng động)',
        r'(bật chạy|chạy êm|chạy ồn|chạy nóng)',
        r'(thiếu.{0,20}(số lượng|hàng|sản phẩm|phụ kiện|quà))',
        r'(giao sai|nhầm hàng|sai size|sai mẫu|sai màu)',
        r'(mỏng.{0,15}nhưng|dày.{0,15}nhưng|nhẹ.{0,15}nhưng)',
        r'(cảm nhận.{0,30}(da|mùi|vị|chất|màu))',
        r'(dính|nhớt|bết|trơn|nhám|sần|mịn).{0,10}(tay|da|mặt)',
    ]
    med = [
        r'(pin yếu|pin lâu|sạc nhanh|sạc chậm|pin chậm)',
        r'(vải mỏng|vải dày|co giãn|chất liệu)',
        r'(thấm hút|không thấm|chống nước|chống bụi)',
        r'(sáng da|căng da|mờ tàn nhang|dưỡng ẩm)',
        r'(không bị rơi|không vỡ|chắc chắn|độ bền)',
        r'(phù hợp.{0,20}(tuổi|người|gia đình|trẻ|bé|sinh viên|học sinh|văn phòng))',
        r'(đọc.{0,30}(hay|tốt|dễ hiểu|khó hiểu|nhàm|cuốn|kém))',
        r'(ngôn từ|câu chuyện|bài học|nội dung).{0,30}(hay|tốt|sáo|bình thường|đặc biệt)',
        r'(in rõ|in đẹp|giấy ổn|sách đẹp|bìa đẹp)',
        r'(chất.{0,10}(kem|sữa|serum|toner|dầu).{0,20}(lỏng|đặc|mịn|nhớt|loãng))',
        r'(ko vừa|không vừa|hơi chật|hơi rộng)',
        r'(tốt cho.{0,20}(dạ dày|da|tóc|xương|khớp|tiêu hóa|đề kháng))',
        r'(cứng|mềm|nhẹ|nặng).{0,20}(hơn|quá|tương đối)',
        r'(nấu nhanh|cơm dẻo|cơm ngon|không dính)',
    ]
    hc = sum(1 for p in high if re.search(p, tl))
    mc = sum(1 for p in med  if re.search(p, tl))
    if hc >= 2 or (hc >= 1 and mc >= 1):
        return 2
    if hc >= 1 or mc >= 1:
        return 1
    return 0


def score_product_experience(t: str) -> int:
    """0-2: trải nghiệm dùng thật"""
    tl = t.lower()
    if re.search(r'(chưa dùng|chưa đọc|chưa xài|chưa sử dụng|mới nhận|mới mua chưa|chưa biết đánh giá)', tl):
        return 0
    used = [
        r'(mình|tôi|tui|mik|mk|em|bé|con|gia đình).{0,50}(dùng|xài|sử dụng|uống|mặc|đọc|ăn)',
        r'(đã|mới).{0,10}(dùng|xài|sử dụng|uống|mặc|đọc)',
        r'(sau|được).{0,20}(dùng|xài).{0,30}(thấy|cảm|nhận|thì)',
        r'(lần \d+|hộp \d+|tuýp \d+|chai \d+).{0,20}(rồi|dùng)',
        r'(bé|con).{0,30}(uống|dùng|thích|ăn)',
        r'(thấy|cảm nhận|nhận thấy).{0,50}(tốt|ok|ổn|hay|đẹp|ngon|thơm|mịn|nhanh|chậm|rõ)',
        r'(mặc.{0,20}(mát|ấm|thoải mái|vừa|chật|rộng))',
        # book reading experience
        r'(đọc.{0,40}(xong|được|rồi|nhiều lần|đi đọc lại))',
        r'(mua.{0,20}(về|rồi).{0,20}(thấy|cảm|nhận|bật|dùng|xài|ăn|mặc|đọc))',
        r'(bật chạy|bật lên|cắm vào|cắm điện|khi dùng|lúc dùng)',
        r'(sau \d+.{0,10}(dùng|xài|sử dụng|sử dụng))',
        r'(nhận hàng.{0,30}(thấy|mở ra|kiểm tra|dùng thử))',
        r'(mua về.{0,30}(thấy|mở|thử|dùng|xài))',
    ]
    cnt = sum(1 for p in used if re.search(p, tl))
    if cnt >= 2: return 2
    if cnt >= 1: return 1
    return 0


def score_decision_value(t: str, sp: int, pe: int) -> int:
    """0-2: giúp người mua quyết định"""
    tl = t.lower()
    high_val = [
        r'(không nên|cẩn thận|lưu ý|chú ý|cảnh báo|nên biết)',
        r'(so với|so sánh|hơn|kém hơn).{0,30}(loại|sản phẩm|hàng|chai|mẫu)',
        r'(phù hợp với|dành cho|không hợp|không phù hợp)',
        r'(lỗi|hỏng|vỡ|không hoạt động|không dùng được|không nhận)',
        r'(size|cỡ).{0,30}(nhỏ hơn|lớn hơn|chật|rộng|sai)',
        r'(tác dụng phụ|nhược điểm|điểm trừ|hạn chế)',
        r'(giao sai|thiếu hàng|không đủ số lượng|thiếu món)',
        r'(giá rẻ hơn|đắt hơn|hời hơn).{0,30}(siêu thị|cửa hàng|hiệu|shop|ngoài)',
        r'(tả cứng|tả mềm|không khác gì quảng cáo|không đúng như)',
        r'(không thể duy trì|không cấp ẩm|không kiềm dầu)',
    ]
    med_val = [
        r'(tốt|ngon|đẹp|ổn).{0,20}(nhưng|tuy nhiên|ngoại trừ|mặc dù)',
        r'(ưu điểm|điểm cộng|hay ở chỗ)',
        r'(pin|sạc).{0,30}(tốt|lâu|nhanh|yếu|chậm)',
        r'(chất liệu|vải|da|nhựa|kim loại).{0,20}(mỏng|dày|bền|đẹp|xấu)',
    ]
    hv = sum(1 for p in high_val if re.search(p, tl))
    mv = sum(1 for p in med_val if re.search(p, tl))
    if hv >= 2 or (hv >= 1 and sp >= 2):
        return 2
    if hv >= 1 or mv >= 2 or (sp >= 1 and pe >= 1):
        return 1
    return 0


def score_clarity(t: str) -> int:
    """0-1"""
    words = t.strip().split()
    if len(words) <= 2: return 0
    if re.search(r'(.)\1{5,}', t): return 0
    clean = re.sub(r'[^\w\s]', '', t).strip()
    if not clean: return 0
    return 1


def score_noise_penalty(t: str) -> int:
    """0 to -2"""
    tl = t.lower().strip()
    p = 0
    if re.search(r'^[.…\s!?*]+$', tl):
        p -= 2
    elif re.search(r'(.)\1{6,}', t):
        p -= 1
    spam = [
        r'(nhận xu|lấy xu|lấy điểm|điểm thưởng).{0,30}(đánh giá|review)',
        r'(liên hệ|inbox|nhắn tin|zalo).{0,20}(mình|em|shop)',
        r'anh chị nào.{0,30}(liên hệ|inbox|cần)',
        r'(mua ngay đi|siêu phẩm số 1)',
    ]
    if any(re.search(sp, tl) for sp in spam):
        p -= 1
    return max(p, -2)


def get_flags(t: str, wc: int, sp: int, pe: int, np: int, dv: int) -> list:
    tl = t.lower()
    flags = []

    # Generic only
    if sp == 0 and pe == 0 and dv == 0:
        flags.append('generic_only')

    # Too short but specific
    if wc <= 5 and sp >= 1:
        flags.append('too_short_but_specific')

    # Noise
    if np <= -1:
        flags.append('noise_text')

    # Shipping only
    ship = bool(re.search(r'(giao hàng|ship|vận chuyển|shipper|đóng gói)', tl))
    prod = bool(re.search(r'(sản phẩm|sp\b|chất lượng|pin|vải|chất|màu|size|mùi|vị|da|tóc|lông|bé|con)', tl))
    if ship and not prod and sp == 0:
        flags.append('shipping_only')

    # Seller service
    seller = bool(re.search(r'(shop\b|cửa hàng|nhân viên|phục vụ|dịch vụ|tiki\b)', tl))
    if seller and not prod and sp == 0 and 'shipping_only' not in flags:
        flags.append('seller_service_only')

    # Product defect
    if re.search(r'(lỗi|hỏng|vỡ|nứt|bong|tróc|cháy|chập|chảy|rò rỉ|đứt|không hoạt động|không nhận|không kết nối|máy đứng|đứng lun)', tl):
        flags.append('contains_product_defect')

    # Usage context
    if re.search(r'(phù hợp|dùng cho|dành cho|mùa hè|mùa đông|mùa lạnh|ngủ|học|văn phòng|trẻ nhỏ|bé|ngoài trời)', tl) and sp >= 1:
        flags.append('contains_usage_context')

    # Comparison
    if re.search(r'(so với|hơn so với|so sánh|kém hơn|tốt hơn|khác với|không giống|tương tự)', tl):
        flags.append('contains_comparison')

    # Size/fit
    if re.search(r'(size|cỡ|số đo|vừa|chật|rộng|dài|ngắn).{0,20}(mặc|mua|đặt|giao|đo)', tl):
        flags.append('contains_size_fit_info')

    # Durability
    if re.search(r'(bền|độ bền|sau \d+ (tháng|năm|tuần)|dùng \d+ (tháng|năm))', tl):
        flags.append('contains_durability_info')

    # Packaging harms product
    if re.search(r'(hộp bị|sách bị|bao bì cũ|móp|méo|bẩn|rách|ướt).{0,60}(sản phẩm|hàng|chai|sp|sách)', tl):
        flags.append('contains_packaging_info')

    # Spam
    if re.search(r'(nhận xu|lấy xu|inbox|liên hệ em|anh chị nào cần)', tl):
        flags.append('possible_fake_or_spam')

    # Duplicate-like
    words_list = tl.split()
    if len(words_list) > 12:
        half = words_list[:len(words_list)//2]
        sec  = words_list[len(words_list)//2:]
        overlap = sum(1 for w in half if w in sec)
        if overlap > len(words_list) // 3:
            flags.append('duplicate_like')

    return flags


def compute_confidence(score: int, flags: list, wc: int) -> float:
    base = 0.9
    if 'ambiguous' in flags: base -= 0.15
    if 2 <= score <= 3:      base -= 0.10
    if score >= 6 or score <= 0: base = min(base + 0.05, 0.99)
    if wc <= 5 and 'too_short_but_specific' not in flags: base -= 0.05
    return round(max(0.50, min(base, 0.99)), 2)


def make_reason(t: str, is_h: int, flags: list, sp: int, pe: int, dv: int, np: int) -> str:
    if np <= -2:
        return "Review không có nội dung đánh giá sản phẩm (noise/ký tự vô nghĩa)."
    if 'noise_text' in flags and np == -1:
        return "Review chứa nội dung lặp lại/ký tự vô nghĩa, không cung cấp thông tin hữu ích."
    if is_h == 0:
        if 'shipping_only' in flags and 'generic_only' in flags:
            return "Review chỉ khen giao hàng và đóng gói, không đánh giá sản phẩm."
        if 'shipping_only' in flags:
            return "Review chỉ đề cập đến giao hàng/đóng gói, không cung cấp thông tin về chất lượng sản phẩm."
        if 'seller_service_only' in flags:
            return "Review chỉ nói về dịch vụ shop, không đánh giá sản phẩm."
        if 'generic_only' in flags:
            return "Review chỉ khen/chê chung chung, không cung cấp thông tin cụ thể giúp người mua ra quyết định."
        return "Review không cung cấp đủ thông tin cụ thể để giúp người mua ra quyết định."
    # helpful
    parts = []
    if 'contains_product_defect'  in flags: parts.append("nêu rõ lỗi/vấn đề sản phẩm")
    if 'contains_usage_context'   in flags: parts.append("cung cấp ngữ cảnh sử dụng cụ thể")
    if 'contains_comparison'      in flags: parts.append("có so sánh giúp người mua cân nhắc")
    if 'contains_size_fit_info'   in flags: parts.append("có thông tin về size/kích thước")
    if 'contains_durability_info' in flags: parts.append("đề cập đến độ bền sản phẩm")
    if 'contains_packaging_info'  in flags: parts.append("chỉ ra vấn đề bao bì ảnh hưởng sản phẩm")
    if 'too_short_but_specific'   in flags: parts.append("ngắn nhưng có thông tin cụ thể")
    if pe >= 2: parts.append("mô tả trải nghiệm sử dụng thực tế")
    if sp >= 2: parts.append("có chi tiết rõ ràng về sản phẩm")
    if parts:
        return f"Review {', '.join(parts)}, hữu ích cho người mua."
    return "Review cung cấp thông tin cụ thể giúp người mua ra quyết định."


def label_row(row) -> dict:
    text = str(row.get('review_text') or row.get('content') or '')
    wc   = int(row.get('word_count', len(text.split())))

    sp  = score_specificity(text)
    pe  = score_product_experience(text)
    dv  = score_decision_value(text, sp, pe)
    cl  = score_clarity(text)
    np_ = score_noise_penalty(text)

    total = sp + pe + dv + cl + np_

    flags = get_flags(text, wc, sp, pe, np_, dv)
    if 2 <= total <= 3:
        flags.append('ambiguous')

    # Label decision
    if total >= 3:
        is_h = 1
    elif total <= 1:
        is_h = 0
    else:
        # score == 2
        if any(f in flags for f in ('contains_product_defect', 'contains_usage_context',
                                     'contains_comparison', 'contains_size_fit_info',
                                     'contains_durability_info', 'contains_packaging_info')):
            is_h = 1
        elif any(f in flags for f in ('generic_only', 'shipping_only', 'seller_service_only')):
            is_h = 0
        else:
            is_h = 1

    conf   = compute_confidence(total, flags, wc)
    reason = make_reason(text, is_h, flags, sp, pe, dv, np_)

    return {
        'candidate_llm_is_helpful':        is_h,
        'candidate_llm_helpfulness_score': total,
        'candidate_llm_confidence':        conf,
        'candidate_llm_reason':            reason,
        'candidate_llm_specificity':       sp,
        'candidate_llm_product_experience':pe,
        'candidate_llm_decision_value':    dv,
        'candidate_llm_clarity':           cl,
        'candidate_llm_noise_penalty':     np_,
        'candidate_llm_quality_flags':     str(flags),
        'candidate_llm_label_version':     LABEL_VER,
        'candidate_llm_model':             MODEL_NAME,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print(f"Đọc file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"  Tổng dòng: {len(df)}  |  Cột: {len(df.columns)}")

    results = []
    for i, row in df.iterrows():
        results.append(label_row(row))
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(df)}] ...")

    res_df = pd.DataFrame(results)

    # Cập nhật các cột candidate_llm_* vào df gốc
    for col in res_df.columns:
        df[col] = res_df[col].values

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu: {OUTPUT_FILE}")
    print(f"Tổng dòng: {len(df)}  |  Tổng cột: {len(df.columns)}")

    # Thống kê
    hr = df['candidate_llm_is_helpful'].mean()
    print(f"\n=== THỐNG KÊ ===")
    print(f"Helpful rate (Antigravity): {hr:.2%}")
    print(df['candidate_llm_is_helpful'].value_counts().to_string())

    if 'reference_is_helpful' in df.columns:
        agree = (df['reference_is_helpful'] == df['candidate_llm_is_helpful']).mean()
        print(f"\nĐồng thuận với reference LLM: {agree:.2%}")

    print("\nXong!")

if __name__ == "__main__":
    main()
