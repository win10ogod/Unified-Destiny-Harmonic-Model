# 統一命理諧振模型（Unified Destiny Harmonic Model）

## 目標

本專案不是把八字、易經、六爻、數字學並排相加，而是把它們轉換為單一可計算的數學狀態空間，建立一個可執行、可回測、可比較的統一模型。

模型核心做法：

1. **八字 / 四柱**
   - 年、月、日、時四柱全部轉成干支週期索引。
   - 月柱使用 **24 節氣** 分段。
   - 五行向量由天干與地支藏干權重組成。

2. **易經 / 六爻**
   - 六條爻不再用擲錢隨機生成。
   - 改用「個體狀態 × 時間狀態」投影成六維爻位勢能。
   - 當某爻位勢能接近零且局部變化率升高時，判定為**動爻**。
   - 六爻正負號轉成陰陽 bit，組成本卦；動爻翻轉後得到之卦。

3. **數字學**
   - Life Path、Attitude、Birthday、Expression 轉為數值循環相位。
   - 保留 11 / 22 / 33 Master Number 的權重特性。

4. **連續時間層**
   - 加入太陽黃經與月相角，形成連續的時間場。
   - 使模型可對任意日期連續計算，而不是只能做離散占例。

5. **統一分數**
   - 干支相位對齊
   - 五行結構對齊
   - 數字學相位對齊
   - 卦象結構對齊
   - 連續天文相位對齊
   - 波動懲罰

最後輸出 `0~100` 的統一分數，以及本卦、之卦、動爻、主通道。

---

## 這個模型有什麼是創新的

創新點不在於新增術語，而在於**把不同命理系統視為不同週期空間的同一個投影問題**：

- 干支：60 週期複數嵌入
- 數字學：9 週期複數嵌入
- 五行：5 維元素空間
- 六爻：5 維元素經固定矩陣投影到 6 維爻位勢能
- 卦變：由位勢零點與導數共同決定，而非隨機 toss

這使模型同時保有：

- **數學上的連續性**
- **六爻的離散卦變機制**
- **四柱的時序結構**
- **數字學的週期映射**

---

## 誠實的可驗證性界線

這份程式**不是**在宣稱「已被科學證明能真實預測未來」。

本程式提供兩種驗證：

1. **內建 benchmark**
   - 驗證模型是否比基線模型有更高的結構分辨率與卦型覆蓋。
   - 指標包括：
     - `signature_dispersion`
     - `score_std`
     - `distinct_signature_ratio`
     - `hexagram_variety`

2. **CSV 外部回測**
   - 使用你自己的真實結果資料驗證預測力。
   - 會輸出：
     - Pearson correlation
     - Spearman correlation
     - Directional accuracy

只有 **CSV 外部回測** 才能判斷模型對真實事件是否有預測價值。

---

## 檔案說明

- `unified_destiny_model_gui.py`：主程式，含 GUI / CLI / 回測 / benchmark
- `hexagrams_compact.json`：64 卦資料集，含中文卦名、英文名、binary、Wilhelm line text
- `requirements.txt`：Python 套件依賴

---

## 執行環境

建議：

- Python 3.11+
- 作業系統：Windows / macOS / Linux

安裝依賴：

```bash
pip install -r requirements.txt
```

若系統沒有 `tkinter`：

- Windows / macOS 的官方 Python 通常已內建
- Linux 可能需要額外安裝系統套件，例如 `python3-tk`

---

## GUI 啟動

```bash
python unified_destiny_model_gui.py
```

打開後可使用四個頁籤：

- **未來一週**：輸入出生資料，計算未來一週
- **基準測試**：執行內建 benchmark
- **CSV 回測**：對接真實結果資料
- **理論**：查看模型定義與驗證邏輯

---

## CLI 用法

### 1. 未來一週預測

```bash
python unified_destiny_model_gui.py --nogui --demo --birth "1990-01-01 12:34" --timezone Asia/Taipei --longitude 121.5654 --start-date 2026-04-15 --days 7
```

### 2. 內建 benchmark

```bash
python unified_destiny_model_gui.py --nogui --benchmark
```

### 3. CSV 回測

```bash
python unified_destiny_model_gui.py --nogui --backtest your_data.csv --birth "1990-01-01 12:34"
```

---

## CSV 格式

至少兩欄：

```csv
date,outcome
2026-04-01T12:00:00+08:00,63.5
2026-04-02T12:00:00+08:00,47.2
2026-04-03T12:00:00+08:00,81.1
```

規則：

- `date`：ISO 8601 時間字串
- `outcome`：你定義的真實結果分數

`outcome` 可以是：

- 每日情緒量表
- 每日交易績效
- 每日工作完成度
- 每日健康 / 睡眠品質分數
- 每日社交互動品質分數

前提是：**定義必須固定一致**。

---

## 參數說明

- `--birth`：出生時間，例如 `1990-01-01 12:34`
- `--name`：姓名，給數字學 Expression Number 使用，可留空
- `--timezone`：IANA 時區，例如 `Asia/Taipei`
- `--longitude`：經度，用於真太陽時修正
- `--start-date`：預測起始日
- `--days`：預測天數
- `--no-true-solar`：停用真太陽時
- `--no-zi-rollover`：停用 23:00 換日

---

## 工程細節

### 四柱計算

- 年柱、月柱、日柱、時柱皆可計算
- 月柱依節氣切換
- 時柱依日干對照規則換算

### 卦象生成

- 六條爻由連續位勢函數生成
- 無需擲銅錢或蓍草
- 可重現、可批次、可回測

### 分數模型

輸出以下分數：

- `bazi_score`
- `hex_score`
- `num_score`
- `naive_score`
- `unified_score`

其中 `unified_score` 是本模型的主分數。

---

## 已完成的本地驗證

在本交付版本中，已完成：

- Python 語法檢查
- CLI 預測實跑
- 內建 benchmark 實跑
- CSV 回測入口實跑
- `tkinter` 模組載入檢查

---

## 你接下來要做的事

若你要真正判斷這個模型是否有外部預測力，直接做三件事：

1. 固定一個出生資料
2. 連續收集 60~180 天真實結果 `outcome`
3. 用 `--backtest` 跑相關與方向準確率

這一步不是可選項，而是唯一能檢驗預測性的步驟。
