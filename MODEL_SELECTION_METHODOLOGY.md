# หลักการเลือกและเปรียบเทียบโมเดล — Baseline, Core Forecasting, Cold-Start

> เอกสารอ้างอิงเพื่อทำความเข้าใจ ไม่ใช่ deliverable ของโปรเจกต์ — สรุปจากบทสนทนาระหว่างพัฒนา Phase 6, 7, 10

## ภาพรวม

โปรเจกต์นี้แยกงานพยากรณ์ยอดขายออกเป็น 3 phase ที่มีจุดประสงค์ต่างกัน แต่ละ phase เลือกโมเดล/วิธีมาเปรียบเทียบโดยมี
**หลักการที่ตั้งใจไว้ชัดเจน** ไม่ใช่การสุ่มเลือกโมเดลมาลองหลายๆ ตัวแล้วดูว่าตัวไหนตัวเลขดีที่สุด

| Phase | คืออะไร | ตอบคำถามอะไร |
|---|---|---|
| Baseline (Phase 6) | เกณฑ์อ้างอิงขั้นต่ำ ไม่มีการเรียนรู้จากข้อมูลเลย | "ถ้าไม่ทำอะไรเลย แม่นแค่ไหน" |
| Core Forecasting (Phase 7) | โมเดลพยากรณ์หลักของโปรเจกต์ (use case 1) | "โมเดลจริงจังดีกว่า baseline แค่ไหน และเพราะอะไร" |
| Cold-Start (Phase 10) | พยากรณ์สินค้าที่ไม่มีประวัติการขาย (use case 4) | "ไม่มีประวัติ จะพยากรณ์ยังไง และแม่นยำไล่ทันสินค้าเก่าเร็วแค่ไหน" |

---

## 1. Baseline (Phase 6)

### หลักการเลือก
ต้องครอบคลุม **สมมติฐานง่ายๆ ที่ต่างกัน** เกี่ยวกับยอดขาย โดยไม่มีการเรียนรู้จากข้อมูลเลยแม้แต่น้อย — เลือก 3 ตัวให้แทน
มุมมองคนละแบบ ไม่ใช่สุ่มเลือกมา:

| วิธี | สมมติฐาน |
|---|---|
| Naive | อนาคต = ปัจจุบัน (ไม่มีอะไรเปลี่ยน) |
| Seasonal Naive | ฤดูกาลซ้ำรอยเดิมทุกปี |
| Moving Average (4w) | แนวโน้มระยะสั้นล่าสุดคือตัวบ่งชี้ที่ดีที่สุด |

**เหตุผลที่ต้องมีครบ 3 มุมมอง**: เพื่อให้มั่นใจว่า "เกณฑ์ขั้นต่ำ" ไม่ได้ต่ำเกินจริงจากการเลือกวิธีที่แย่แบบไม่ยุติธรรม
ถ้าเลือกแค่ Naive ตัวเดียวมาเป็นเกณฑ์ อาจดูเหมือนโมเดลจริงจังเก่งเกินจริง

### วิธีเปรียบเทียบ
- ไม่มีการ "เลือกผู้ชนะเก็บไว้ใช้งานจริง" — แค่หาว่าตัวไหนดีที่สุดในกลุ่มนี้ (Moving Average, WAPE 0.243)
- ใช้ตัวเลขนั้นเป็น **"ราคาที่ต้องเอาชนะ"** สำหรับ Core Forecasting

---

## 2. Core Forecasting (Phase 7)

### หลักการเลือก
**ไม่ได้เลือกโมเดลมาลองสุ่มๆ** แต่ออกแบบให้ทดสอบ **แกนการออกแบบที่แยกจากกันได้** ในการทดลองเดียว:

| แกนที่ 1: pooling ข้อมูล | แกนที่ 2: วิธีการ | แกนที่ 3: library (boosting vs bagging) |
|---|---|---|
| Global (โมเดลเดียวรวมทุกสินค้า) vs Local (แยกต่อสินค้า) | LightGBM (Machine Learning) vs Holt-Winters ETS (สถิติคลาสสิก) | LightGBM vs XGBoost vs Random Forest vs CatBoost (ทั้ง 4 ตัว global pooled, ฟีเจอร์/fold ชุดเดียวกัน) |

**เหตุผลของการออกแบบแบบนี้**: ทำให้รู้ได้ว่า **"อะไรกันแน่"** ที่ทำให้ผลดีขึ้น ไม่ใช่แค่รู้ว่า "โมเดล A ดีกว่า B" เฉยๆ
เช่น การที่ Global LightGBM ชนะ Local LightGBM (อัลกอริทึมเดียวกัน ต่างแค่วิธีจัดกลุ่มข้อมูล) พิสูจน์ได้ชัดว่า
**การ pooling ข้อมูลข้ามสินค้าช่วยจริง** ไม่ใช่แค่ LightGBM เก่งกว่า ETS เฉยๆ

**แกนที่ 3 (library) เพิ่มเข้ามาทีหลัง** ในฐานะ **robustness/challenger check** ไม่ใช่ส่วนหนึ่งของการออกแบบ 2 แกนดั้งเดิม:
Global pooled XGBoost (`reg:absoluteerror`, ฟีเจอร์/categorical handling/fold ชุดเดียวกับ Global pooled LightGBM
ทุกประการ), Global pooled CatBoost (`loss_function='RMSE'` เหมือน L2 ของ LightGBM, `depth=6` — จัดการ categorical ด้วย
ordered target encoding ของตัวเอง ส่งคอลัมน์ categorical เป็น string ผ่าน `cat_features`) และ Global pooled Random Forest (`sklearn.RandomForestRegressor` — bagging แทน boosting, ต้อง
ordinal-encode categorical columns เพราะ sklearn RF ไม่รองรับ category dtype แบบ native เหมือน LightGBM/XGBoost
ส่วนค่า `promo_recency` ที่เป็น NaN เติมด้วย sentinel เป็นทางเลือกเชิงโมเดลเพื่อให้ต้นไม้แยกแถวเหล่านี้ออกจากค่า
recency จริงได้ ไม่ใช่ข้อจำกัดของไลบรารี — sklearn เวอร์ชันที่ pin ไว้ (1.9) รองรับ NaN โดยตรงอยู่แล้วตั้งแต่ 1.4)
ใช้ตรวจว่าผลของ Global pooled LightGBM ไม่ได้ดีเพราะบังเอิญ
เจาะจงกับ implementation ตัวเดียว — เทียบราย fold พร้อม standard deviation และ paired significance test (ไม่ใช่แค่
ค่าเฉลี่ยตัวเดียว) ดู [`07_core_forecasting.ipynb`](notebooks/01_forecasting/07_core_forecasting.ipynb) ส่วนที่ 2-4

**หมายเหตุ**: `min_child_samples=20` (LightGBM), `min_samples_leaf=20` (Random Forest) และ `min_child_weight=20`
(XGBoost) ตั้งเลขเดียวกันเพื่อให้ "ความพยายาม regularize" ใกล้เคียงกัน ไม่ใช่เพราะทั้งสามค่ามีความหมายเดียวกันทุก
ประการ — CatBoost ไม่ได้ตั้งค่านี้เพราะต้นไม้แบบ symmetric ที่เป็นค่า default ไม่มีพารามิเตอร์ขนาด leaf ขั้นต่ำ (ใช้ได้เฉพาะ `grow_policy` แบบ Depthwise/Lossguide) จึงคงรูปแบบต้นไม้ default ไว้ — `min_samples_leaf` ของ Random Forest นับจำนวนแถวตรงๆ แบบ exact ส่วน `min_child_samples` ของ LightGBM
เป็นเป้าหมายจำนวนแถวเชิงประมาณเท่านั้น (เอกสารของ LightGBM เองระบุว่าคำนวณแบบ approximation จาก Hessian ทำให้
บาง leaf อาจมีแถวน้อยกว่าค่านี้ได้) ส่วน `min_child_weight` ของ XGBoost คือ threshold บนผลรวม Hessian (second
derivative) ของแถวใน leaf ซึ่งเท่ากับจำนวนแถวจริงก็ต่อเมื่อทุกแถวมี Hessian = 1 (เป็นจริงสำหรับ squared-error loss
แต่ไม่ได้การันตีสำหรับ `reg:absoluteerror` ที่ใช้อยู่ที่นี่) ดูรายละเอียดใน comment ที่ `src/models/forecast.py`

### วิธีเปรียบเทียบ
1. WAPE บน walk-forward CV fold **ชุดเดียวกัน** ทุกโมเดล (7 fold)
2. Final holdout (10 สัปดาห์สุดท้าย) แยกต่างหากจาก CV average — ทำแล้วใน Phase 13 (fit ครั้งเดียว ไม่ปรับอะไรจากผล): Global LightGBM WAPE 0.2238 บน 9 สัปดาห์ที่ label ครบ เทียบกับ CV 0.2235 (สัปดาห์สุดท้ายมี label เพียง 2/7 วันจึงแยกออก) ดู [`13_future_holdout_backtest.ipynb`](notebooks/01_forecasting/13_future_holdout_backtest.ipynb)
3. **เช็ค feature importance เพิ่ม** เพื่อยืนยันว่าโมเดลที่ชนะเรียนรู้อะไรที่สมเหตุสมผลจริง ไม่ใช่แค่ตัวเลขต่ำเพราะบังเอิญ
4. **แกน library**: เทียบราย fold + standard deviation ระหว่าง LightGBM, XGBoost, Random Forest, CatBoost — ทดสอบนัยสำคัญ
   (paired t-test ข้าม fold) ของ **LightGBM (โมเดลหลัก) เทียบกับ challenger ทั้งสามตัวแยกกัน** คือ LightGBM vs
   XGBoost, LightGBM vs Random Forest และ LightGBM vs CatBoost ไม่ใช่แค่เลือกคู่ที่ WAPE เฉลี่ยต่ำสุด 2 อันดับมาทดสอบคู่เดียว
   (ซึ่งเมื่อมีหลายโมเดล ไม่จำเป็นต้องเป็นคู่ที่ใกล้กันที่สุดจริงๆ) — เพราะทดสอบ 3 คู่พร้อมกันจึงปรับ p-value ด้วย
   **Holm correction** ด้วย เพื่อคุมอัตรา false positive รวม (วิธีเดียวกับที่ใช้ใน Phase 11)

### ผลลัพธ์
WAPE เฉลี่ยของ 4 libraries อยู่ในช่วงแคบมาก — Global Pooled CatBoost 0.2215, LightGBM 0.2235, Random Forest 0.2241,
XGBoost 0.2249 (ห่างกันสูงสุด 0.0034) CatBoost ได้ค่าเฉลี่ยต่ำสุดทั้ง WAPE, MAE และ RMSE แต่ผลต่างยังไม่ผ่านนัยสำคัญหลัง
ปรับ p-value (ดูด้านล่าง) จึงยังคงเลือก **Global Pooled LightGBM เป็น "โมเดลหลัก"** ที่ใช้ต่อใน Phase 10, 11 และ 11b
เพราะ phase เหล่านั้นสร้างบนโมเดลนี้อยู่แล้ว — ไม่ใช่เพราะแม่นยำกว่า

Paired t-test ของ LightGBM เทียบกับ challenger ทั้งสามตัว (n = 7 folds, Holm-corrected 3 คู่):

| คู่เปรียบเทียบ | mean WAPE diff | 95% CI | raw p | Holm-adjusted p | มีนัยสำคัญที่ 0.05? |
|---|---|---|---|---|---|
| LightGBM vs XGBoost | −0.0014 | [−0.0035, 0.0007] | 0.149 | 0.297 | ไม่ |
| LightGBM vs Random Forest | −0.0006 | [−0.0031, 0.0020] | 0.606 | 0.606 | ไม่ |
| LightGBM vs CatBoost | +0.0020 | [0.0004, 0.0037] | 0.024 | 0.072 | ไม่ (เฉียด) |

ไม่พบความแตกต่างที่มีนัยสำคัญทางสถิติกับ challenger ตัวใด (ใช้คำว่า "ไม่พบความแตกต่างที่มีนัยสำคัญ" อย่างตั้งใจ
แทน "พิสูจน์แล้วว่าเท่ากัน" — การไม่ reject H0 ไม่ใช่หลักฐานว่าไม่มีความต่างเลย เพียงแต่ข้อมูล 7 folds ไม่พอจะสรุปว่าต่าง
อย่างมีนัยสำคัญ) กรณี CatBoost ควรอ่านอย่างระมัดระวังเป็นพิเศษ: raw p = 0.024 ต่ำกว่า 0.05 และ CatBoost ได้ WAPE ต่ำกว่า LightGBM
ใน 5 จาก 7 folds แต่หลัง Holm correction เหลือ 0.072 จึงตีความได้เพียงว่ามี **แนวโน้ม** ดีกว่าเล็กน้อย (ประมาณ 0.9% ในเชิง
สัมพัทธ์) ยังไม่ใช่ข้อสรุปที่ยืนยันได้ ผลลัพธ์โดยรวมจึง**สอดคล้องกับ**การที่ผลลัพธ์ robust ข้าม library — ไม่ใช่การ
"พิสูจน์" ความเท่ากัน (การจะอ้างแบบนั้นได้ต้องกำหนด equivalence margin ไว้ล่วงหน้า ซึ่งการทดลองนี้ไม่ได้ทำ) ขณะที่ความต่างระหว่าง
Global pooled กับ Local per-SKU LightGBM (WAPE 0.257) ใหญ่กว่าความต่างระหว่าง library ราวสิบเท่า XGBoost, Random Forest และ CatBoost
ไม่ได้ทำหน้าที่เป็นเพดานอ้างอิงหรือถูก carry ต่อไปยัง phase อื่น — หากต้องการเปลี่ยนโมเดลหลักเป็น CatBoost ต้อง re-run
Phase 10, 11 และ 11b ใหม่ทั้งหมด

**หมายเหตุเรื่อง reproducibility**: ตัวเลขข้างต้นมาจากการรันแบบ deterministic (LightGBM ตั้ง `deterministic=True` +
`force_row_wise=True`, Random Forest รันแบบ single-thread `n_jobs=1`) เพราะพบว่า fixed `random_state` อย่างเดียว
ไม่พอทำให้ผลลัพธ์เหมือนกันทุกครั้งที่รันใหม่เมื่อมี multithreading — ยืนยันแล้วว่ารันซ้ำ 2 รอบใน environment เดียวกัน
ให้ผลเหมือนกันทุกบิต ตัวเลขชุดนี้ (อัปเดตล่าสุด) รันบน environment ที่ตรงกับ `requirements-lock.txt` จริง ณ ตอนอัปเดต
(`lightgbm==4.6.0`, `numpy==2.0.2`, `pandas==2.3.3`, Python 3.9.6, macOS) — ตัวเลขจาก commit ก่อนหน้านี้ผลิตจาก
environment อื่น (`lightgbm==4.7.0`, `numpy==2.5.2`, `pandas==3.0.5`, Python 3.13.0, Windows) จึงคลาดเคลื่อนจาก
ตัวเลขชุดนี้ไปเล็กน้อยในทศนิยมตำแหน่งที่ 3-4 — เป็น library-version drift ข้าม environment ไม่ใช่ความไม่เสถียรของโค้ด

**อัปเดตเมื่อเพิ่ม CatBoost**: ตัวเลขของ Phase 7 ในเอกสารนี้ (ทั้ง 4 libraries) รันใหม่ใน `.venv` ปัจจุบันบน Windows
(Python 3.13.0, LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.0, CatBoost 1.2.10) — คือ environment ชุดเดียวกับที่ย่อหน้าก่อนระบุว่า
คลาดเคลื่อนเล็กน้อยจาก lock file ไม่ใช่ environment ของ `requirements-lock.txt` (lock file เพิ่มเฉพาะ CatBoost และ dependency ยังไม่ได้
regenerate) ทั้ง 4 libraries มาจากการรันครั้งเดียวกันจึงเทียบกันได้ตรงไปตรงมา แต่ต่าง environment จากตัวเลขของ Phase 10, 11 และ 11b
ผลของการเปลี่ยน environment: mean WAPE ของ LightGBM (0.2235) และ Random Forest (0.2241) ไม่เปลี่ยน แต่ WAPE ราย fold ของ
LightGBM ขยับเล็กน้อย และ XGBoost เปลี่ยนจาก 0.2240 เป็น 0.2249 ส่วน p-value ก่อนหน้า (p ≈ 0.801) ใช้ไม่ได้อีกเพราะจำนวนการทดสอบ
เพิ่มจาก 2 เป็น 3 คู่ CatBoost ตั้ง `random_seed=42` และตรวจแล้วว่ารันซ้ำและเปลี่ยนจำนวน thread (1 กับ 4) ได้ผลเหมือนกันทุกบิตทั้ง 7 folds

---

## 3. Cold-Start (Phase 10)

แยกสองสถานการณ์ เพราะข้อมูลที่ทราบ ณ วันพยากรณ์และ target ไม่เหมือนกัน:

| สถานการณ์ | วิธีที่เปรียบเทียบ | ข้อมูล / เป้าหมาย |
|---|---|---|
| ก่อนเริ่มขาย | Category mean, Analog ไม่ใช้ราคา, metadata-only LightGBM | ไม่มี target-SKU sales; ทำนาย recorded units 7 วันแรก |
| หลังเริ่มขาย | Analog, Meta-core, Full-core | เมื่อจบสัปดาห์ปัจจุบัน ทำนายสัปดาห์ถัดไป |

Meta-core ตัด lag/rolling แต่ยังใช้ยอดขายสัปดาห์ปัจจุบัน จึงไม่ใช่ zero-history Full-core เพิ่มประวัติเท่าที่มีจริงและไม่ใช่เพดาน accuracy ของ mature SKU; ทั้งคู่ใช้ LightGBM

ก่อนเริ่มขาย category mean ใช้หมวด/channel/region เดียวกัน พร้อม fallback ที่บันทึกไว้ Analog เลือกสินค้าต้นแบบจาก metadata โดยไม่ใช้ราคาที่เกิดหลังเปิดตัว ทุกวิธีใช้เฉพาะ launch labels ที่เกิดครบก่อน cutoff

คง 5 SKU เดิมเป็นชุดอ้างอิง เพิ่ม 15 SKU ที่มีประวัติเปิดตัวก่อนหน้าอย่างน้อย 10 SKU ตามวันและจำนวนเท่านั้น กัน 5 SKU เดิมออกจาก train ทุกครั้ง ไม่ tune บนผลทดสอบ รวม 20 SKU/180 แถว รายงาน extension แยกจากผลรวม

ผลรวม WAPE: Category mean 30.27%, LightGBM 32.10%, Analog 35.87% แต่ LightGBM ต่ำสุดบน extension 15 SKU และ paired SKU-bootstrap ของ Category mean − LightGBM คร่อมศูนย์ จึงใช้ค่าเฉลี่ยเป็น baseline หลัก ไม่อ้าง superiority ทั่วไป

วันเปิดตัวและ assortment อนุมานย้อนหลังจากข้อมูลจำลอง สมมติว่าผู้วางแผนรู้ล่วงหน้า ไม่ใช่ verified launch plan กราฟตามอายุเป็น descriptive ยังไม่ระบุสัปดาห์ที่ตามทัน mature products ได้ และห้ามเทียบคะแนนข้ามสองสถานการณ์ตรง ๆ

ดู protocol, ผลครบ, exploratory uncertainty, ข้อจำกัด และการทำซ้ำใน [รายงานปิดงาน](reports/phase10/final_summary.md)

---

## สรุปเปรียบเทียบทั้ง 3 Phase

| Phase | หลักการเลือกตัวเปรียบเทียบ | มิติที่ใช้ตัดสิน |
|---|---|---|
| Baseline | ครอบคลุมสมมติฐานง่ายๆ ที่ต่างกัน 3 แบบ | ตัวเลขเฉลี่ยตัวเดียว (หาเกณฑ์ขั้นต่ำ) |
| Core Forecasting | แยกทดสอบ pooling × วิธีการ + เช็ค library axis (LightGBM vs XGBoost vs Random Forest vs CatBoost) เพิ่ม | WAPE (CV + final holdout ใน Phase 13) + ความสม่ำเสมอ + feature importance + std dev ราย fold + paired significance test (library axis) |
| Cold-Start | แยก pre-launch และ short-history พร้อม baseline ที่เรียบง่าย | แถวร่วม/coverage, ผลตาม cohort/อายุ และ exploratory SKU uncertainty |

## หลักการร่วมที่ใช้ทุก Phase (ไม่เปลี่ยนแปลง)

1. **ข้อมูล/metric เดียวกันทุกโมเดลเสมอ** — ห้ามโมเดลไหนได้เปรียบจากข้อมูลพิเศษ
2. **แบ่งข้อมูลตามเวลาเสมอ** (walk-forward CV) — ไม่มีโมเดลไหนเห็นอนาคตตอนเทรน
3. **ไม่เชื่อแค่ตัวเลขเฉลี่ย** — ต้องดูความสม่ำเสมอ/รายละเอียดเพิ่มเสมอ ก่อนสรุปว่าโมเดลไหนดีที่สุด
