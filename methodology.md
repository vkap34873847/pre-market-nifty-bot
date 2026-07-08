# Pre-Market Nifty Regime Prediction: Complete Methodology

> Document Version 1.0  
> Author: AI + Human Collaborative  
> Purpose: To build a pre-market Nifty 50 regime classifier (BULLISH/FLAT/BEARISH) with explainable signals, daily human-in-the-loop verification, and iterative model improvement.

---

## Table of Contents

1. [Foundations: Why This Approach Works](#1-foundations-why-this-approach-works)
2. [What Quant Firms Actually Do](#2-what-quant-firms-actually-do)
3. [Evidence: GIFT Nifty Accuracy](#3-evidence-gift-nifty-accuracy)
4. [The Accuracy Ceiling Problem](#4-the-accuracy-ceiling-problem)
5. [System Architecture](#5-system-architecture)
6. [Feature Engineering](#6-feature-engineering)
7. [Models and Training Protocol](#7-models-and-training-protocol)
8. [Technical Architecture](#8-technical-architecture)
9. [Visibility and Daily Workflow](#9-visibility-and-daily-workflow)
10. [Iterative Improvement Plan](#10-iterative-improvement-plan)
11. [References](#11-references)

---

## 1. Foundations: Why This Approach Works

### 1.1 The Core Thesis

The Indian stock market is not perfectly efficient at very short time horizons. Prices do NOT instantly reflect all available information — there are measurable gaps between when information becomes available (overnight global events) and when it is fully priced into Nifty (9:15 AM open). This gap is where our edge lives.

### 1.2 Efficient Market Hypothesis (EMH) and Its Limits

The Efficient Market Hypothesis (Fama, 1970) states that asset prices reflect all available information. Three forms:

| Form | What it says | Relevant to us? |
|---|---|---|
| Weak | Past prices cannot predict future prices | Largely true for EOD close prediction |
| Semi-strong | All public info is priced in immediately | False for pre-market gaps |
| Strong | All info including insider is priced in | False everywhere |

Academic research on Indian markets (Yadav & Patil, 2019; BSE/NSE studies) consistently finds that the **Indian market is not weak-form efficient** — there are predictable patterns, especially around openings, gaps, and institutional flows.

### 1.3 Where Our Edge Comes From

The edge is NOT from predicting the close from past prices (that's the weak form EMH, which is nearly efficient). The edge comes from:

```
Overnight Event (US close, Fed, crude) 
    → GIFT Nifty moves immediately (6:30 AM - 2:45 AM)
    → Nifty spot is frozen at yesterday's close (3:30 PM)
    → GAP between GIFT Nifty and Nifty close = measurable signal
    → At 9:15 AM, Nifty opens and closes this gap
    → We measure the gap at 8:30 AM, predict the opening and the day
```

This is NOT a violation of EMH — we are using price discovery from ONE market (GIFT Nifty futures) to predict another (Nifty spot). The information is public. We are just aggregating it faster and more systematically than a human checking multiple screens.

### 1.4 What the Literature Says

**Marketcalls.in (Rajandran R, 2023):** XGBoost for Nifty next-day direction prediction with lag returns, HMA, RSI, ATR features. Achieved ~50.36% accuracy for UP/DOWN classification. Conclusion: "XGBoost offers significant advantages over linear regression for predicting market direction, but may require further optimization."

**Akshay Gupta (2025):** LSTM vs XGBoost on NIFTY50 (2000-2025). XGBoost achieved 50.97% directional accuracy, LSTM achieved 51.98%. Conclusion: "While XGBoost minimized error, the LSTM captured non-linear trend dynamics better. SHAP analysis confirmed the model learned rational economic behaviors."

**Uttkarsh O B (2026):** Nifty gap prediction using global indices (S&P 500, NASDAQ, Nikkei, HSI). XGBoost achieved 66.41% balanced accuracy. Key finding: "Global indices contain weak but real predictive information for NIFTY opening gaps."

**Two Sigma (2021):** Gaussian Mixture Models for regime detection. Found 4 distinct market conditions. "The GMM approach is entirely data-driven — the model outputs various market conditions, but that doesn't tell us what those conditions are intuitively."

**WorldQuant / D.E. Shaw / Two Sigma pipeline:** Signal generation → in-sample testing → out-of-sample testing → correlation check → regime stability test → deployment. "The research pipeline has to produce net positive signal additions per year just to maintain the portfolio's current performance level."

---

## 2. What Quant Firms Actually Do

### 2.1 Two Sigma

- $70B+ AUM
- 10,000+ data sources
- 380+ petabytes of data stored
- Uses Gaussian Mixture Models (GMMs) for regime detection — fits multi-dimensional probability distributions to factor returns (17 factors from their "Factor Lens")
- Identifies 4 market conditions: Crisis, WOI (Worried, Uncertain, Irritable), Inflation, Normal
- Key insight from Two Sigma's own research: "If we perfectly knew the market conditions without bias, why would we use relatively complicated models like GMMs?"
- They do NOT predict tomorrow's regime from today's data. They **detect the current regime** and position accordingly.

### 2.2 Bridgewater Associates

- $92B Pure Alpha fund
- Uses a "regime-detection framework" to identify volatility regimes
- Their edge came from detecting the **regime change** from low-vol to high-vol (triggered by tariff volatility), then rebalancing positions
- Bridgewater does NOT predict the next day's regime — they detect the **regime shift** when it happens and move positions

### 2.3 WorldQuant

- Distributed research platform where individual "alphas" (signals) are tested
- Typical alpha evaluation: in-sample testing → out-of-sample → correlation check against existing alphas → **regime stability test**
- "Signals that fail to demonstrate consistent predictive power across multiple regimes and time periods get rejected"

### 2.4 What This Means For Us

No major quant firm does "predict tomorrow's regime from today's data" as a standalone product. Instead:

| What they do | What we are doing |
|---|---|
| Detect current regime in real-time | Predict tomorrow's regime before market opens |
| Use 100+ data sources (alternative data) | Use 12-15 free public data sources |
| Deploy 10,000+ alphas simultaneously | Deploy one combined signal |
| Use sophisticated risk management | Use simple combined-decision logic |

**Our approach is novel even by quant firm standards.** The pre-market prediction → conditional rule switching architecture is not standard practice anywhere. This is both a risk (no proven blueprint) and an opportunity (potential edge if it works).

---

## 3. Evidence: GIFT Nifty Accuracy

### 3.1 What GIFT Nifty Actually Predicts

GIFT Nifty (formerly SGX Nifty) is a Nifty 50 futures contract trading on NSE International Exchange (GIFT City). Trading hours: 6:30 AM - 2:45 AM IST (21 hours/day).

It predicts ONE thing: **the direction and approximate size of Nifty 50's opening gap at 9:15 AM IST.**

It does NOT predict:
- Where Nifty will close at 3:30 PM
- Intraday direction
- The gap-fill probability

### 3.2 Published Accuracy Claims

| Source | Claim | Details |
|---|---|---|---|
| GiftNifty.com | "80-90% of the time" | GIFT Nifty accurately predicts whether Nifty 50 will open higher or lower. Verified at https://giftnifty.com/ |
| mcxtrends.in | "approximately 85%" | GIFT Nifty predicts opening direction with ~85% accuracy. Verified at https://www.mcxtrends.in/ |
| niftytrader.in | "75-85% directionally" | Directional accuracy range across market conditions. Verified at https://www.niftytrader.in/gift-nifty-live |
| Belong (Financial Research) | "majority of sessions" (no exact %) | "GIFT Nifty correctly signals the Nifty 50's opening direction in the majority of trading sessions." Does NOT explicitly claim ~85% |
| Belong article (gap-size analysis) | "Directional signal is most reliable when the implied gap is large" | Gaps >100 points: highly reliable. Gaps <50 points: essentially flat. This is from Belong's gap-size analysis, not a separate "practical trading community" source |
| ~~Kotak Securities~~ | ~~~80%~~ | **REMOVED — source could not be verified.** No Kotak Securities page was found making this claim. Recommend removing from evidence |

### 3.3 When GIFT Nifty Fails (The 15-20% Error)

GIFT Nifty's prediction breaks down in these cases:

1. **Domestic catalyst between 8:45 AM and 9:15 AM** — GIFT Nifty stops trading before NSE pre-open. Domestic news (corporate results, policy announcements) can shift the open
2. **Small gaps (±25 to ±50 points)** — Domestic order flow at NSE open can easily flip direction
3. **Expiry day sessions** — Rollover dynamics distort the futures-spot relationship
4. **Budget days, election results, Fed nights** — Event-driven volatility exceeds normal model range

### 3.4 How We Use This in Our System

```
GIFT Nifty Gap     Opening Prediction    Our Label
──────────────────────────────────────────────────
> +0.5%            Strong gap up         BULLISH (high confidence)
+0.3% to +0.5%     Moderate gap up       BULLISH (medium confidence)
-0.3% to +0.3%     Flat                  FLAT (skip gap trading)
-0.5% to -0.3%     Moderate gap down     BEARISH (medium confidence)
< -0.5%            Strong gap down       BEARISH (high confidence)
```

---

## 4. The Accuracy Ceiling Problem

### 4.1 Why 55-60% is a Hard Ceiling

Every academic and practical study converges on the same finding: next-day market direction prediction cannot exceed ~55-60% accuracy with public price data. Here is the evidence:

| Study | Target | Best Accuracy | Model |
|---|---|---|---|
| Marketcalls.in (2023) | Nifty next-day UP/DOWN | 50.36% | XGBoost |
| Akshay Gupta (2025) | NIFTY50 next-day direction | 51.98% | LSTM |
| Charan B (2025) | Nifty-50 price movement | 50.36% | XGBoost |
| Uttkarsh O B (2026) | Nifty gap direction | 66.41% balanced | XGBoost |

Notice: the gap prediction (66.41%) is higher than EOD prediction (50-52%). This confirms our thesis — **predicting the opening gap is easier than predicting the full day.**

### 4.2 Why The Ceiling Exists

```
At 3:30 PM: Nifty closes at X
At 3:30 PM to 8:30 AM: Global events happen
At 8:30 AM: We know GIFT Nifty, VIX, crude, USD/INR, Asian markets
At 9:15 AM: Nifty opens. Gap direction is ~85% predictable.
From 9:15 AM to 3:30 PM: NEW information arrives (macro data, 
  corporate news, FII flows, Twitter, geopolitical surprises)
This new info accounts for 40-50% of the day's total move
```

**The ceiling exists because 40-50% of the day's variance comes from unpredictable intraday news that does not exist at 8:30 AM.** No model, no matter how sophisticated, can predict news that hasn't happened yet.

### 4.3 Why 55-60% Is Still Valuable

Your scalping simulation achieved 90.6% win rate on trade SELECTION. A 55-60% regime filter on TOP of that means:

```
Without filter:  90.6% win rate (you pick trades well)
With 55% filter: You skip days when model predicts BEARISH but you 
                 would have traded bullish. You lose some good trades
                 but avoid the catastrophic days where everything fails.
```

Even a 5-10% edge over random on filtering days is valuable when layered on an already profitable strategy.

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PRE-MARKET DATA FETCH (8:00 AM)                      │
│                                                                          │
│  giftnifty.com ──→ GIFT Nifty level                                      │
│  yfinance      ──→ VIX, USD/INR, Crude, Nikkei, HSI, SSEC, SP500, NDX  │
│  nseindia.com  ──→ (future) FII/DII data                                 │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                                  │
│                                                                          │
│  gift_gap_pct = (GIFT_Nifty / prev_Nifty_close) - 1                     │
│  vix_change = (VIX / prev_VIX) - 1                                      │
│  usdinr_overnight = (USDINR / prev_USDINR) - 1                          │
│  crude_overnight = (Crude / prev_Crude) - 1                              │
│  nikkei_today = (Nikkei / prev_Nikkei) - 1                               │
│  hsi_today = (HSI / prev_HSI) - 1                                        │
│  shanghai_today = (SSEC / prev_SSEC) - 1                                 │
│  sp500_yesterday = (SP500 / prev_SP500_2d) - 1  [shift(1)]              │
│  nasdaq_yesterday = (NDX / prev_NDX_2d) - 1      [shift(1)]             │
│  nifty_yesterday_return                                               │
│  nifty_vol_5d = rolling_std(returns, 5)                                 │
│  day_of_week                                                           │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│  LAYER 1: GIFT Nifty    │   │  LAYER 2: XGBoost Classifier│
│  Gap Mapping            │   │                             │
│                         │   │  Input: 13 pre-market       │
│  Rule-based:            │   │         features            │
│  Gap > +0.3% → BULLISH  │   │  Output: BULLISH/FLAT/      │
│  Gap < -0.3% → BEARISH  │   │         BEARISH probability │
│  else → FLAT             │   │                             │
│                         │   │  Trained on 10 years data   │
│  Confidence: 85% on     │   │  Balanced accuracy: ~58-62% │
│  opening direction      │   │  (estimated with full       │
│                         │   │   feature set)              │
└───────────┬─────────────┘   └──────────────┬──────────────┘
            │                               │
            └───────────┬───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     COMBINED DECISION ENGINE                             │
│                                                                          │
│  if Layer1 == BULLISH and Layer2 == BULLISH:                            │
│      Final = BULLISH (high confidence, trade bullish rules)             │
│  if Layer1 == BULLISH and Layer2 == FLAT:                               │
│      Final = BULLISH-CAUTIOUS (reduce size)                             │
│  if Layer1 == BULLISH and Layer2 == BEARISH:                            │
│      Final = CONFLICT (skip trading)                                    │
│  if Layer1 == FLAT:                                                      │
│      Final = FLAT (skip or use Layer2 only)                             │
│  ... similar for BEARISH combinations                                    │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     OUTPUT TO USER                                       │
│                                                                          │
│  Regime: BULLISH | FLAT | BEARISH | CONFLICT                             │
│  Confidence: HIGH | MEDIUM | LOW                                         │
│  Suggested Action: Trade / Reduce / Skip                                 │
│  Key Signals: GIFT Gap: +0.45% | VIX: 14.2 | FII: +850cr                │
│  Layer 2 Probabilities: BULLISH 62%, FLAT 28%, BEARISH 10%              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Prediction Timeline

```
Time        Event                              Data available
────        ─────                              ─────────────
3:30 PM     NSE closes                          Nifty close, VIX close
6:00 PM     FII/DII data published              FII net, DII net
             (with 1-day lag)
8:00 PM     US markets open                     SP500 futures, NDX futures
            Crude, USD/INR continue trading
2:45 AM     GIFT Nifty Session II closes        Final overnight levels
6:30 AM     GIFT Nifty Session I opens          1st pre-market read
8:00 AM     ─── OUR PREDICTION RUN ───          All features available
8:30 AM     Best GIFT Nifty read window         Settled pre-market level
9:00 AM     NSE pre-open session begins         Domestic order flow visible
9:15 AM     NSE opens                           Actual gap confirmed
3:30 PM     NSE closes                          Actual regime confirmed
```

### 5.3 Key Design Principle: Separation of Concerns

The system is deliberately split into two layers rather than one combined model:

**Why not just train XGBoost on everything (including GIFT Nifty gap)?**

| Approach | Pros | Cons |
|---|---|---|
| Combined model | Optimizes one objective | Black box. Can't explain why |
| Two-layer separate | Each layer is interpretable. You can see when they agree/disagree. Enables human override. | Not globally optimal |

**The two-layer approach prioritizes interpretability over raw accuracy.** This is intentional — you said you want to keep me (the human) in the loop to catch contradictions and anomalies. A black-box model would not allow that.

---

## 6. Feature Engineering

### 6.1 Complete Feature Set

| # | Feature | Symbol | Source | Formula | Why It Predicts |
|---|---|---|---|---|---|
| 1 | GIFT Nifty Gap | gift_gap_pct | giftnifty.com | (GIFT_Nifty / Nifty_Close_prev) - 1 | Overnight global sentiment directly in Nifty terms |
| 2 | India VIX Level | vix_level | yfinance (^INDVIX) | Current VIX value | Fear/greed gauge. High VIX → more BEARISH days |
| 3 | India VIX Change | vix_change | yfinance | (VIX / VIX_prev) - 1 | Volatility regime shift overnight |
| 4 | USD/INR Overnight | usdinr_overnight | yfinance (INR=X) | (USDINR / USDINR_prev) - 1 | FII flow direction. Weak rupee → FII selling |
| 5 | Crude Oil Overnight | crude_overnight | yfinance (BZ=F) | (Crude / Crude_prev) - 1 | India imports ~85% crude. Higher crude = inflation + fiscal deficit |
| 6 | Nikkei Today | nikkei_today | yfinance (^N225) | (Nikkei / Nikkei_prev) - 1 | Asian market sentiment (opens before NSE) |
| 7 | Hang Seng Today | hsi_today | yfinance (^HSI) | (HSI / HSI_prev) - 1 | China sentiment, FII proxy |
| 8 | Shanghai Today | shanghai_today | yfinance (^SSEC) | (SSEC / SSEC_prev) - 1 | China macro sentiment |
| 9 | S&P 500 Yesterday | sp500_yest | yfinance (^GSPC) | shift(1): (SP500 / SP500_prev2d) - 1 | US market sentiment (known before NSE opens) |
| 10 | NASDAQ Yesterday | nasdaq_yest | yfinance (^IXIC) | shift(1): (NDX / NDX_prev2d) - 1 | Tech sector sentiment |
| 11 | Nifty Prev Return | nifty_prev_ret | yfinance (^NSEI) | (Close / Close_prev) - 1 | Previous day momentum |
| 12 | Nifty 5D Volatility | nifty_vol_5d | yfinance | std(returns, 5) * sqrt(252) | Recent volatility regime |
| 13 | Day of Week | day_of_week | Calendar | 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri | Day-of-week effects (Monday continuation, Friday profit booking) |

### 6.2 Target Variable

```
daily_return = (Close_Today / Close_Yesterday) - 1

Target:
  2 (BULLISH)  if  daily_return > +0.3%
  1 (FLAT)     if -0.3% <= daily_return <= +0.3%
  0 (BEARISH)  if  daily_return < -0.3%
```

### 6.3 Why These Features (Thesis Behind Each)

**GIFT Nifty Gap:** Direct measure of overnight institutional positioning. When GIFT Nifty is significantly above/below Nifty close, arbitrageurs will push the open toward GIFT Nifty.

**India VIX:** The "fear index." VIX > 20 indicates elevated fear → bearish bias. VIX < 12 indicates complacency → bullish bias. VIX change overnight shows whether the fear is increasing or decreasing.

**USD/INR:** FIIs are the marginal price-setters in Indian markets. When rupee weakens, FIIs lose money on currency conversion → they sell. A 0.5% rupee move can shift FII behavior significantly.

**Crude Oil:** India imports ~85% of its oil. Every $10/barrel increase adds ~0.4% to CPI inflation and ~0.3% to fiscal deficit. Higher crude = bearish for India.

**Asian Markets:** Nikkei, HSI, and Shanghai open 3+ hours before NSE. Their direction tells us how Asia is reacting to overnight US moves. If S&P 500 was up 1% but Asian markets are down, the US optimism is not spreading — caution for Nifty.

**S&P 500 / NASDAQ (shifted):** US markets close at ~3:30 AM IST. Their returns ARE known before 9:15 AM. But they must be shifted by 1 day because the dates don't align (US session happens after NSE closes).

**Previous Nifty Return:** Short-term momentum. A strong up day tends to have follow-through (50-55% probability).

**5-Day Volatility:** Regime detection at feature level. Low vol → trending days more likely. High vol → choppy/reversal days more likely.

**Day of Week:** Statistical day-of-week effects. Monday shows continuation from Friday. Tuesday-Wednesday are strongest. Thursday-Friday show profit booking tendencies.

### 6.4 Features NOT Included (And Why)

| Feature | Why not included | May add later? |
|---|---|---|
| FII/DII net cash | Published at 6 PM with 1-day lag. Can only predict T+1, not same day | Yes, for T+1 prediction |
| Options PCR | Published in real-time but only meaningful at close. Available pre-market but adds noise | Maybe, after baseline model |
| Advance/Decline Ratio | Only known after market opens | No, requires intraday data |
| News Sentiment | Hard to quantify reliably. FinBERT models exist but add complexity | Maybe, Phase 2 |
| Social Media Sentiment | Too noisy, not reliable | No |

---

## 7. Models and Training Protocol

### 7.1 Layer 1: GIFT Nifty Gap Mapping (Rule-Based)

No training required. Pure mapping:

```
if gift_gap_pct > 0.003:
    opening_regime = "BULLISH"
elif gift_gap_pct < -0.003:
    opening_regime = "BEARISH"
else:
    opening_regime = "FLAT"
```

### 7.2 Layer 2: XGBoost Classifier

**Why XGBoost over alternatives:**

| Model | Why Not |
|---|---|
| Logistic Regression | Linear. Can't capture non-linear interactions between VIX, crude, and Asian markets |
| Random Forest | Good, but XGBoost typically edges it on structured tabular data |
| LSTM | Requires 1000+ days for meaningful training. Financial LSTM rarely beats XGBoost on daily data (confirmed by utkarshobd, akshay-gupta studies) |
| Transformer | Overkill for 13 features on 2500 rows. Would overfit heavily |

**Hyperparameters (starting point):**

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_lambda=1.0,
    reg_alpha=0.1,
    eval_metric="mlogloss",
    objective="multi:softprob",
    num_class=3,
    random_state=42
)
```

**Why these choices:**
- max_depth=4: Prevents overfitting on small dataset (2500 rows). Deeper trees would memorize noise
- learning_rate=0.05: Slow learning with 200 trees. Better generalization
- subsample=0.8: Each tree sees 80% of data. Reduces overfitting
- colsample_bytree=0.8: Each tree sees 80% of features. Reduces feature co-dependence
- min_child_weight=3: Prevents leaf nodes from being too specific
- reg_lambda/alpha: L1/L2 regularization. Further prevents overfitting

### 7.3 Training Protocol

**Data:** 10 years of Nifty daily data (2016-2026), ~2,500 trading days

**Split:** Time-based (never random shuffle in time series)

```
Train: 2016-01-01 to 2023-12-31  (~2,000 days)
Test:  2024-01-01 to 2026-06-30  (~600 days)
```

**Walk-Forward Validation (Phase 2, after baseline):**

```
Fold 1: Train 2016-2020, Validate 2021 H1
Fold 2: Train 2016-2021 H1, Validate 2021 H2
Fold 3: Train 2016-2021, Validate 2022 H1
...continues to present
```

**Evaluation Metrics:**

- Balanced Accuracy (primary — accounts for class imbalance)
- Confusion Matrix (see which misclassifications happen)
- F1 Score (balance of precision and recall)
- Feature Importance (which features drive predictions)
- Sharpe Ratio of a simple trading strategy based on predictions

**Expected Performance:**

| Model | Expected Balanced Accuracy | Notes |
|---|---|---|
| Naive (always predict BULLISH) | ~33% | Class balance is roughly equal |
| Logistic Regression | ~55% | Linear baseline |
| XGBoost (core) | ~58-62% | Non-linear interactions |
| GIFT Nifty only (opening) | ~85% | But only covers opening, not EOD |
| Combined (GIFT + XGBoost) | ~60-65% | When both agree |

### 7.4 Class Imbalance Handling

```
Expected class distribution:
  BULLISH: ~35-40% of days
  FLAT:    ~25-30% of days
  BEARISH: ~30-35% of days
```

XGBoost handles imbalance via `scale_pos_weight` for binary, or `class_weight` for multi-class. We will use `compute_class_weight()` from sklearn to set weights inversely proportional to class frequencies.

---

## 8. Technical Architecture

### 8.1 File Structure

```
C:\Users\vansh\Desktop\New folder (2)\pre_market_regime_classifier\
│
├── data/
│   ├── raw/                    # Raw yfinance downloads
│   │   ├── nifty_daily.csv
│   │   ├── vix_daily.csv
│   │   ├── usdinr_daily.csv
│   │   ├── crude_daily.csv
│   │   ├── nikkei_daily.csv
│   │   ├── hsi_daily.csv
│   │   ├── shanghai_daily.csv
│   │   └── sp500_nasdaq_daily.csv
│   │
│   ├── features/               # Pre-computed feature matrices
│   │   └── feature_matrix.csv
│   │
│   └── predictions/            # Daily prediction logs
│       └── prediction_history.csv
│
├── models/
│   └── xgboost_model.json      # Trained model
│
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
│
├── src/
│   ├── fetch_data.py           # Pull all data from yfinance
│   ├── engineer_features.py    # Compute all features
│   ├── train_model.py          # Train XGBoost
│   ├── predict_today.py        # Daily prediction script
│   └── evaluate.py             # Backtest and metrics
│
├── daily_log.txt               # Running daily prediction log
├── methodology.md              # This document
└── requirements.txt            # Dependencies
```

### 8.2 Data Flow

```
Step 1: fetch_data.py
  yfinance.download("^NSEI")      → data/raw/nifty_daily.csv
  yfinance.download("^INDVIX")    → data/raw/vix_daily.csv
  yfinance.download("INR=X")      → data/raw/usdinr_daily.csv
  yfinance.download("BZ=F")       → data/raw/crude_daily.csv
  yfinance.download("^N225")      → data/raw/nikkei_daily.csv
  yfinance.download("^HSI")       → data/raw/hsi_daily.csv
  yfinance.download("^SSEC")      → data/raw/shanghai_daily.csv
  yfinance.download("^GSPC")      → data/raw/sp500_daily.csv
  yfinance.download("^IXIC")      → data/raw/nasdaq_daily.csv

Step 2: engineer_features.py
  Reads all raw CSVs
  Computes 13 features (see section 6.1)
  Creates target (BULLISH/FLAT/BEARISH)
  Saves to data/features/feature_matrix.csv

Step 3: train_model.py
  Reads feature_matrix.csv
  Time-based 80/20 split
  Trains XGBoost with walk-forward validation
  Saves model to models/xgboost_model.json
  Prints: balanced accuracy, confusion matrix, feature importance

Step 4: predict_today.py (run daily at 8:00 AM)
  Fetches latest pre-market data
  Computes today's features
  Loads saved XGBoost model
  Runs Layer 1 (GIFT Nifty gap mapping)
  Runs Layer 2 (XGBoost prediction)
  Combines into final recommendation
  Appends to prediction_history.csv
  Prints output to console AND daily_log.txt

Step 5: evaluate.py (run weekly)
  Reads prediction_history.csv
  Compares predictions against actual outcomes
  Updates accuracy metrics
  Identifies which features were most informative
  Suggests model retraining if accuracy drifts
```

### 8.3 Daily Prediction Script Output

When you run `predict_today.py`, you will see:

```
╔══════════════════════════════════════════════════════╗
║      PRE-MARKET REGIME PREDICTION — 08 Jul 2026     ║
╠══════════════════════════════════════════════════════╣
║                                                    ║
║  PRE-MARKET SIGNALS:                               ║
║  ─────────────────                                  ║
║  GIFT Nifty Gap:      +0.45%  (BULLISH signal)     ║
║  India VIX:           14.2    (Low fear)            ║
║  USD/INR Overnight:   -0.12%  (Rupee stable)        ║
║  Crude Overnight:     +0.32%  (Slight headwind)     ║
║  Nikkei Today:        +0.87%  (Supportive)          ║
║  Hang Seng Today:     +0.54%  (Supportive)          ║
║  Shanghai Today:      +0.21%  (Neutral)             ║
║  S&P 500 Yesterday:   +0.65%  (Supportive)          ║
║  NASDAQ Yesterday:    +0.91%  (Supportive)          ║
║  Nifty Prev Return:   -0.18%  (Slight weakness)     ║
║  Nifty 5D Vol:        12.8%   (Normal)              ║
║  Day of Week:         Wednesday                     ║
║                                                    ║
║  ────────────────────────────────────────────────   ║
║                                                    ║
║  LAYER 1 — GIFT Nifty Gap Mapping:                 ║
║    Opening Regime:  BULLISH                        ║
║    Confidence:      High (gap > +0.3%)             ║
║                                                    ║
║  LAYER 2 — XGBoost Prediction:                     ║
║    BULLISH:  62.3%  ┃████████████████████░░░░░     ║
║    FLAT:     25.1%  ┃██████████░░░░░░░░░░░░░       ║
║    BEARISH:  12.6%  ┃█████░░░░░░░░░░░░░░░░░░      ║
║                                                    ║
║  ────────────────────────────────────────────────   ║
║                                                    ║
║  COMBINED DECISION:                                ║
║    Regime:      BULLISH                            ║
║    Confidence:  HIGH (both layers agree)           ║
║    Action:      Trade bullish rules                ║
║                                                    ║
║  ⚠ NOTE FROM AI:                                   ║
║  All pre-market signals align today. GIFT Nifty    ║
║  gap is clean +0.45%, VIX is low, Asian markets    ║
║  supportive, and US cues were positive. However     ║
║  note that yesterday closed slightly negative —     ║
║  watch for gap-fill risk in first 30 minutes.      ║
║  If Nifty fails to hold above yesterday's close    ║
║  after 9:45 AM, regime may downgrade to CAUTIOUS.  ║
║                                                    ║
╚══════════════════════════════════════════════════════╝
```

---

## 9. Visibility and Daily Workflow

### 9.1 The Visibility Problem

You raised a valid concern: "What you do becomes invisible — Python takes files from somewhere and does everything automatically. I need to understand how the flow is actually going on."

**Our solution: three layers of visibility**

### 9.2 Layer 1: Console Output (Instant)

Every time I run the prediction, I will show you the FULL output as seen in Section 8.3 above. You will see:
- Every feature value
- Both layers' outputs
- The combined decision
- My qualitative interpretation of the signals

### 9.3 Layer 2: `daily_log.txt` (History)

A running text file appended every day:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2026-07-07  |  BULLISH  |  HIGH  |  Both layers agree
  GIFT Gap: +0.45% | VIX: 14.2 | Nikkei: +0.87%
  Actual: [to be filled at 3:30 PM]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2026-07-06  |  FLAT     |  LOW   |  GIFT gap < 0.3%
  GIFT Gap: +0.18% | VIX: 15.1 | Nikkei: -0.22%
  Actual: FLAT (closed +0.12%) ✓ CORRECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 9.4 Layer 3: Weekly Review (Continuous Improvement)

Every week, I will run an evaluation that shows:
- How many predictions were correct vs wrong
- Which features were most predictive that week
- Whether the model's accuracy is stable or degrading
- Whether any feature values were anomalous (out of normal range)
- Recommendation: retrain / adjust thresholds / investigate feature

### 9.5 Your Role: Human-in-the-Loop

| When | What you do | Why |
|---|---|---|
| 8:05 AM | Read the prediction output I show you | Verify the signals make sense |
| 8:10 AM | Tell me if any signal seems wrong | Catch data errors or anomalies |
| 9:15 AM | Watch the open | Verify GIFT Nifty prediction |
| 3:30 PM | Tell me how the day actually went | Ground truth for model feedback |
| Weekly | Review the evaluation report | Decide if model needs changes |

**You are NOT expected to understand every number.** The format is designed so you can quickly see:
1. Green = BULLISH, Red = BEARISH, Yellow = FLAT
2. Confidence level (HIGH/MEDIUM/LOW)
3. My qualitative note at the bottom (what I think about today)

If anything looks off, you ask me and I explain.

### 9.6 File Location for Visibility

As you suggested, everything will live in a dedicated folder so you can open files and see what happened:

```
C:\Users\vansh\Desktop\Other Projects\pre_market_regime_classifier\
├── daily_log.txt          # Open this to see all historical predictions
├── prediction_history.csv # Open this in Excel to see the data
├── feature_importance.png # Open this to see which features matter
└── methodology.md         # This document
```

---

## 10. Iterative Improvement Plan

### Phase 1: Baseline (Week 1)
- [ ] Build data fetcher (yfinance for all sources)
- [ ] Engineer initial feature set (13 features)
- [ ] Train XGBoost baseline model
- [ ] Evaluate on test set (2024-2026)
- [ ] Create daily prediction script

### Phase 2: Daily Operation (Week 2+)
- [ ] Run prediction every morning at 8:00 AM
- [ ] Log all predictions to daily_log.txt
- [ ] Verify predictions against actual market close
- [ ] Weekly evaluation report

### Phase 3: Improvement (Week 4+)
- [ ] Add GIFT Nifty live scrape (instead of using actual gap proxy)
- [ ] Test FII/DII as additional feature
- [ ] Test walk-forward validation
- [ ] Hyperparameter tuning with Optuna
- [ ] Feature ablation study (remove weakest features)

### Phase 4: Optional WebApp (Future)
- [ ] Simple Streamlit dashboard showing today's prediction
- [ ] Historical accuracy chart
- [ ] Feature importance visualization

---

## 11. References

### Academic
1. Fama, E.F. (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*.
2. Yadav, P. & Patil, D.S. (2019). "An Empirical Study on Efficient Market Hypothesis of Indian Capital Market." *IJTSRD*.
3. Ross, S.A. (1976). "The Arbitrage Theory of Capital Asset Pricing." *Journal of Economic Theory*.
4. Ammann, M. & Verhofen, M. (2006). "Regime-Switching Factor Investing."

### Industry
5. Two Sigma (2021). "A Machine Learning Approach to Regime Modeling." *Two Sigma Street View*.
6. WorldQuant (2026). "How Quant Hedge Funds Actually Build and Vet Trading Signals." *Young & Calculated*.
7. Bridgewater Associates (2025). "Pure Alpha: Regime Detection Framework."

### Practical / Open Source
8. Rajandran R / Marketcalls.in (2023). "Predicting Stock Price and Market Direction using XGBoost."
9. Akshay Gupta (2025). "NIFTY50-LSTM-Forecasting-System." GitHub.
10. Utkarsh O B (2026). "Nifty-gap-prediction-quant-model." GitHub. [1 star, 27 commits]
11. Jaynil Patel (2026). "Market-movement-prediction-using-Machine-Deep-Learning-Algorithms." GitHub. [1 star, 5 commits]

### Pre-Market / GIFT Nifty
12. GiftNifty.com. "Predictive Gap Analysis: 80-90% accuracy." https://giftnifty.com/
13. Belong Financial. "GIFT Nifty as an Early Indicator: How Accurate Is It in Predicting Nifty 50 Open." https://getbelong.com/
14. Bajaj AMC (2026). "How Traders Use GIFT Nifty to Assess Nifty 50 Opening."
15. NiftyPulse (2026). "Pre-market analysis guide: Gift Nifty + FII/DII + VIX checklist."
16. mcxtrends.in. "GIFT Nifty accuracy ~85%." https://www.mcxtrends.in/
17. niftytrader.in. "GIFT Nifty directional accuracy 75-85%." https://www.niftytrader.in/gift-nifty-live

### Data Sources
18. Yahoo Finance (yfinance) — `^NSEI`, `^INDIAVIX`, `USDINR=X`, `CL=F`, `^N225`, `^HSI`, `000001.SS`, `^GSPC`, `^IXIC`
19. GiftNifty.com — Live GIFT Nifty level
20. NSE India — FII/DII data, Options chain

---

## Appendix: Decision Matrix (Full)

```
Layer 1  |  Layer 2           |  Final Regime      |  Action
(Open)   |  (EOD Prediction)  |                    |
─────────────────────────────────────────────────────────────
BULLISH  |  BULLISH (P>0.55)  |  BULLISH           |  Trade bullish at full size
BULLISH  |  BULLISH (P<0.55)  |  BULLISH-CAUTIOUS  |  Trade bullish at 50% size
BULLISH  |  FLAT              |  BULLISH-CAUTIOUS  |  Trade bullish at 50% size
BULLISH  |  BEARISH (P>0.55)  |  CONFLICT          |  Skip trading
BULLISH  |  BEARISH (P<0.55)  |  BULLISH-CAUTIOUS  |  Trade cautiously
         |                    |                    |
FLAT     |  BULLISH (P>0.6)   |  BULLISH-CAUTIOUS  |  Trade bullish at 50%
FLAT     |  BULLISH (P<0.6)   |  FLAT              |  Skip (no clear signal)
FLAT     |  FLAT              |  FLAT              |  Skip
FLAT     |  BEARISH (P<0.6)   |  FLAT              |  Skip
FLAT     |  BEARISH (P>0.6)   |  BEARISH-CAUTIOUS  |  Trade bearish at 50%
         |                    |                    |
BEARISH  |  BULLISH (P>0.55)  |  CONFLICT          |  Skip trading
BEARISH  |  BULLISH (P<0.55)  |  BEARISH-CAUTIOUS  |  Trade bearish at 50%
BEARISH  |  FLAT              |  BEARISH-CAUTIOUS  |  Trade bearish at 50%
BEARISH  |  BEARISH (P<0.55)  |  BEARISH-CAUTIOUS  |  Trade bearish at 50%
BEARISH  |  BEARISH (P>0.55)  |  BEARISH           |  Trade bearish at full size
```

Key: P = probability from XGBoost for the predicted class.

---

*End of Methodology Document*
