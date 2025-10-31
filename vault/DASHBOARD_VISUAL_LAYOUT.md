# Prediction Dashboard - Visual Layout Guide

## Complete Layout Description

This document provides a detailed visual description of the Prediction Dashboard layout, including dimensions, colors, and component placement.

---

## Overall Window Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TradeApp - Champiguru                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  [SQLite: path...] [Browse] [Table: ohlcv] [Symbol: BTCUSDT] [TF: 1h]     │
│  [Model: artifacts/model_best.pt...] [Browse Model]                        │
│  [Start Daemon] [Stop Daemon]                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Notebook ──────────────────────────────────────────────────────────────┐ │
│ │ [Preview] [Training] [Backtest] [Audit] [Status] [▶ Predictions Dashboard]│
│ │                                                                          │ │
│ │  (Predictions Dashboard Content - See Below)                            │ │
│ │                                                                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─ Logs ──────────────────────────────────────────────────────────────────┐ │
│ │ [INFO] Dashboard initialized...                                         │ │
│ │ [INFO] Model loaded successfully...                                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Predictions Dashboard Tab - Detailed Layout

### Full Tab Structure (1400x900 pixels typical)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ PREDICTIONS DASHBOARD TAB                                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┬───────────────────────────────────────────────────────────────────┐
│                 │                                                                   │
│  CONTROL PANEL  │              VISUALIZATION AREA                                   │
│   (300px wide)  │                 (1100px wide)                                     │
│                 │                                                                   │
│   [Scrollable]  │  ┌─────────────────────────────────────────────────────────────┐ │
│                 │  │ Multi-Horizon Prediction Fan                                │ │
│                 │  │                                                             │ │
│                 │  │  [Large Matplotlib Canvas - 1080x600]                       │ │
│                 │  │                                                             │ │
│                 │  │  Shows:                                                     │ │
│                 │  │  - Historical price line (black)                            │ │
│                 │  │  - Prediction fan (color gradient)                          │ │
│                 │  │  - Confidence bands (shaded)                                │ │
│                 │  │  - Probability layers (violet violins)                      │ │
│                 │  │                                                             │ │
│                 │  └─────────────────────────────────────────────────────────────┘ │
│                 │                                                                   │
│                 │  ┌─────────────────────────────────────────────────────────────┐ │
│                 │  │ METRICS PANEL (Prediction Summary Table)                    │ │
│                 │  │ [Horizon][Target Time][Pred Price][Change][%][CI][Signal]  │ │
│                 │  │   1h      10:30         $51,234    +$234  +0.46%  ...   ↑  │ │
│                 │  │   5h      14:30         $51,890    +$890  +1.75%  ...   ↑  │ │
│                 │  │  10h      19:30         $50,123    -$877  -1.73%  ...   ↓  │ │
│                 │  │  ...                                                        │ │
│                 │  └─────────────────────────────────────────────────────────────┘ │
│                 │                                                                   │
└─────────────────┴───────────────────────────────────────────────────────────────────┘
```

---

## Control Panel - Detailed Breakdown (Left Side, 300px x ~800px scrollable)

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  PREDICTION DASHBOARD          ┃  ← Title (Arial 12pt bold)
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌────────────────────────────────┐
│ Model Configuration            │  ← LabelFrame
├────────────────────────────────┤
│ [Load Model from Artifacts]    │  ← Button (fill X)
│ [Load Model from File...]      │  ← Button (fill X)
│                                │
│ Model loaded                   │  ← Label (green text)
│ Seq Len: 32                    │
│ Horizon: 10                    │
│ Hidden: 64                     │
│ Features: 39                   │
└────────────────────────────────┘

┌────────────────────────────────┐
│ Horizons (Hours)               │  ← LabelFrame
├────────────────────────────────┤
│ ☑ 1h                           │  ← Checkbuttons (vertical)
│ ☑ 3h                           │
│ ☑ 5h                           │
│ ☑ 10h                          │
│ ☐ 15h                          │
│ ☑ 20h                          │
│ ☑ 30h                          │
│                                │
│ [Select All]                   │  ← Button (fill X)
│ [Clear All]                    │  ← Button (fill X)
└────────────────────────────────┘

┌────────────────────────────────┐
│ Visualization                  │  ← LabelFrame
├────────────────────────────────┤
│ ☑ Show Historical Data         │  ← Checkbuttons
│ ☑ Show Confidence Bands        │
│ ☑ Show Probability Layers      │
│                                │
│ Color Scheme:                  │  ← Label
│ [viridis          ▼]           │  ← OptionMenu
└────────────────────────────────┘

┌────────────────────────────────┐
│ Multiple Scenarios             │  ← LabelFrame
├────────────────────────────────┤
│ ☐ Enable Multiple Scenarios    │  ← Checkbutton
│ (Bull/Base/Bear volatility)    │  ← Label (gray, small font)
└────────────────────────────────┘

┌────────────────────────────────┐
│ Update Settings                │  ← LabelFrame
├────────────────────────────────┤
│ ☑ Auto-Refresh                 │  ← Checkbutton
│                                │
│ Interval (minutes):            │  ← Label
│ [━━━━━━●━━━━━━━━━━━━] 5        │  ← Scale (1-60)
└────────────────────────────────┘

┌────────────────────────────────┐
│ Actions                        │  ← Frame (padded)
├────────────────────────────────┤
│ [    Update Now    ]           │  ← Button (green bg, white fg)
│ [   Clear Plot     ]           │  ← Button
│ [   Save Config    ]           │  ← Button
│ [Export Predictions...]        │  ← Button
└────────────────────────────────┘

┌────────────────────────────────┐
│ Status                         │  ← LabelFrame
├────────────────────────────────┤
│ Current Price:                 │  ← Label
│ $51,234.56                     │  ← Label (Arial 14pt bold, blue)
│                                │
│ Last Update:                   │  ← Label
│ 2025-10-30 14:23:45            │  ← Label (green/gray)
│                                │
│ Status:                        │  ← Label
│ Predictions updated            │  ← Label (green/gray/red)
└────────────────────────────────┘
```

---

## Visualization Area - Matplotlib Canvas (Center-Right, 1080x600)

### Prediction Fan Chart Example

```
Price ($)
    │
55k ┤                                    ╱╱╱╱╱ 30h (lightest yellow)
    │                                  ╱╱╱╱╱
    │                                ╱╱╱╱╱ 20h (yellow-green)
54k ┤                              ╱╱╱╱╱
    │                            ╱╱╱╱╱ 15h (green)
    │                          ╱╱╱╱╱
53k ┤                        ╱╱╱╱╱ 10h (teal)
    │          ────────────╱╱╱╱╱
    │      ────────────  ╱╱╱╱╱ 5h (blue)
52k ┤  ────────────    ╱╱╱╱╱
    │──────────      ╱╱╱╱╱ 3h (dark blue)
    │████████     ╱╱╱╱╱ 1h (purple, darkest)
51k ┤████████▓▓▓▓▓▓
    │████████▓▓▓
    │         │              ◐        ◐       ◐  ← Probability violins
50k ┤─────────┼──────────────┼────────┼───────┼─────────►
    10:00   12:00         14:00    16:00   18:00   20:00  Time

Legend:
──── Historical Close (black, bold)
╱╱╱  Prediction lines (color gradient: purple → yellow)
▓▓▓  Confidence bands (shaded, gradient alpha)
◐    Probability density layers (purple violins)
```

**Visual Elements:**

1. **Historical Data (Black Line)**
   - Color: `#000000` (black)
   - Width: 2px
   - Alpha: 0.7
   - Zorder: 10 (on top)

2. **Prediction Lines (Gradient)**
   - 1h: `#440154` (dark purple)
   - 3h: `#31688e` (blue)
   - 5h: `#35b779` (teal)
   - 10h: `#6ece58` (green)
   - 15h: `#b5de2b` (yellow-green)
   - 20h: `#fde724` (yellow)
   - 30h: `#ffffbf` (pale yellow)
   - Width: 1.8px → 1.3px (decreases with horizon)
   - Alpha: 0.8 → 0.5 (fades with horizon)

3. **Confidence Bands (Shaded)**
   - Color: Same as prediction line
   - Alpha: 0.15 → 0.05 (gradient, fades with horizon)
   - Zorder: 2 (behind lines)

4. **Probability Layers (Violet Violins)**
   - Color: `#9932CC` (purple)
   - Alpha: 0.3
   - Displayed at 3 key horizons (e.g., 5h, 10h, 20h)
   - Width: ~3% of x-axis range
   - Zorder: 3

5. **Grid**
   - Color: Gray
   - Alpha: 0.3
   - Style: Dashed (`--`)
   - Width: 0.5px

6. **Axes**
   - X-axis: Datetime ("%Y-%m-%d %H:%M")
   - Y-axis: Price ($)
   - Font: 11pt
   - Labels: 12pt
   - Title: 13pt bold

---

## Metrics Panel - Treeview Table (Bottom, 1080x150)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Prediction Summary                                                              │
├─────────┬──────────────────┬──────────────┬──────────┬──────────┬──────────────┬────────┤
│ Horizon │ Target Time      │ Predicted    │ Change   │ Change   │ 95% CI       │ Signal │
│         │                  │ Price        │ ($)      │ (%)      │              │        │
├─────────┼──────────────────┼──────────────┼──────────┼──────────┼──────────────┼────────┤
│  1h     │ 2025-10-30 11:00 │ $51,234.56   │ +$234.56 │ +0.46%   │ $50,890 -    │   ↑    │
│         │                  │              │          │          │ $51,578      │        │
├─────────┼──────────────────┼──────────────┼──────────┼──────────┼──────────────┼────────┤
│  3h     │ 2025-10-30 13:00 │ $51,567.23   │ +$567.23 │ +1.12%   │ $50,123 -    │   ↑    │
│         │                  │              │          │          │ $53,011      │        │
├─────────┼──────────────────┼──────────────┼──────────┼──────────┼──────────────┼────────┤
│  5h     │ 2025-10-30 15:00 │ $51,890.45   │ +$890.45 │ +1.75%   │ $49,567 -    │   ↑    │
│         │                  │              │          │          │ $54,213      │        │
├─────────┼──────────────────┼──────────────┼──────────┼──────────┼──────────────┼────────┤
│  10h    │ 2025-10-30 20:00 │ $50,123.78   │ -$876.22 │ -1.73%   │ $47,890 -    │   ↓    │
│         │                  │              │          │          │ $52,357      │        │
├─────────┼──────────────────┼──────────────┼──────────┼──────────┼──────────────┼────────┤
│  20h    │ 2025-10-31 06:00 │ $52,456.12   │ +$1,456  │ +2.86%   │ $48,123 -    │   ↑    │
│         │                  │              │          │          │ $56,789      │        │
├─────────┼──────────────────┼──────────────┼──────────┼──────────┼──────────────┼────────┤
│  30h    │ 2025-10-31 16:00 │ $53,234.67   │ +$2,234  │ +4.39%   │ $46,890 -    │   ↑    │
│         │                  │              │          │          │ $59,579      │        │
└─────────┴──────────────────┴──────────────┴──────────┴──────────┴──────────────┴────────┘
                                                                       [▲] Scrollbar
```

**Column Widths:**
- Horizon: 80px (centered)
- Target Time: 150px (centered)
- Predicted Price: 120px (centered)
- Change ($): 120px (centered)
- Change (%): 120px (centered)
- 95% CI: 120px (centered)
- Signal: 60px (centered)

**Colors:**
- Header: Light gray background, bold text
- Rows: Alternating white/light gray
- Positive changes: Green text (#008000)
- Negative changes: Red text (#FF0000)
- Neutral: Black text

**Signals:**
- ↑ (U+2191): Green, if change > +0.5%
- ↓ (U+2193): Red, if change < -0.5%
- → (U+2192): Gray, if -0.5% ≤ change ≤ +0.5%

---

## Color Schemes

### Default Scheme: Viridis

```
Horizon  Color       Hex       RGB
─────────────────────────────────────
  1h     Purple    #440154   (68, 1, 84)
  3h     Blue      #31688e   (49, 104, 142)
  5h     Teal      #35b779   (53, 183, 121)
 10h     Green     #6ece58   (110, 206, 88)
 15h     Y-Green   #b5de2b   (181, 222, 43)
 20h     Yellow    #fde724   (253, 231, 36)
 30h     P-Yellow  #ffffbf   (255, 255, 191)
```

### Alternative: Plasma

```
Horizon  Color       Hex       RGB
─────────────────────────────────────
  1h     D-Purple  #0d0887   (13, 8, 135)
  3h     Purple    #7e03a8   (126, 3, 168)
  5h     Magenta   #cc4778   (204, 71, 120)
 10h     Orange    #f89540   (248, 149, 64)
 15h     Yellow    #fdc527   (253, 197, 39)
 20h     L-Yellow  #f0f921   (240, 249, 33)
 30h     P-Yellow  #fcffa4   (252, 255, 164)
```

### Alternative: Coolwarm (Diverging)

```
Horizon  Color       Hex       RGB
─────────────────────────────────────
  1h     D-Blue    #3b4cc0   (59, 76, 192)
  3h     Blue      #6788ee   (103, 136, 238)
  5h     L-Blue    #9abbff   (154, 187, 255)
 10h     White     #dddddd   (221, 221, 221)
 15h     Pink      #f3a583   (243, 165, 131)
 20h     Orange    #dd7755   (221, 119, 85)
 30h     Red       #b40426   (180, 4, 38)
```

---

## Multiple Scenarios Layout

When "Enable Multiple Scenarios" is checked:

```
Price ($)
    │
55k ┤                    ╱╱╱╱╱ Bear 30h (red gradient)
    │          ─────────╱╱╱╱╱──── Base 30h (blue gradient)
    │   ╱╱╱╱╱──────────        ──── Bull 30h (green gradient)
54k ┤ ╱╱╱╱╱
    │╱╱╱╱╱
53k ┤──── Historical
    │
52k ┤
    │
51k ┤
    │
50k ┤─────────────────────────────────────────►
    10:00           14:00           18:00    Time

Legend:
──── Historical (black)
╱╱╱  Bull scenario (green shades)
╱╱╱  Base scenario (blue shades)
╱╱╱  Bear scenario (red shades)
```

**Color Families:**

**Bull Scenario (Greens):**
- 1h: `#006400` (dark green)
- 10h: `#32CD32` (lime green)
- 30h: `#90EE90` (light green)

**Base Scenario (Blues):**
- 1h: `#00008B` (dark blue)
- 10h: `#4169E1` (royal blue)
- 30h: `#87CEEB` (sky blue)

**Bear Scenario (Reds):**
- 1h: `#8B0000` (dark red)
- 10h: `#DC143C` (crimson)
- 30h: `#FFA07A` (light salmon)

---

## Responsive Behavior

### Window Resize

- **Control Panel**: Fixed width (300px), full height
- **Visualization Area**: Expands to fill remaining width
- **Metrics Panel**: Fixed height (150px), full width
- **PanedWindow**: User can drag divider to adjust control panel width

### Small Screens (<1200px width)

- Control panel scrollable (vertical scroll)
- Plot scales down (maintains aspect ratio)
- Metrics table scrollable (horizontal scroll if needed)

### Large Screens (>1600px width)

- Plot expands to use extra space
- Metrics table columns widen proportionally
- Control panel stays 300px (doesn't expand)

---

## Accessibility

### Color Blindness Support

**Deuteranopia (Red-Green):**
- Use "Coolwarm" or "Plasma" schemes (avoid greens)
- Ensure sufficient contrast (WCAG AA)

**Protanopia (Red-Green):**
- Use "Viridis" or "Plasma" (naturally friendly)
- Line thickness differentiates horizons

**Tritanopia (Blue-Yellow):**
- Use "Plasma" (purple-orange gradient)
- Avoid "Viridis" (blue-yellow gradient)

### High Contrast Mode

- Increase line widths by 1.5x
- Increase font sizes by 2pt
- Darker grid lines (alpha 0.5 → 0.7)
- Thicker axes (2px → 3px)

---

## Animation & Interaction (Future)

### Planned Features

**Hover Tooltips:**
- Hover over prediction line → show exact values
- Hover over confidence band → show probability
- Hover over violin → show distribution stats

**Click Interactions:**
- Click horizon legend → toggle that horizon on/off
- Click prediction point → highlight in metrics table
- Right-click → context menu (export, zoom, etc.)

**Zoom/Pan:**
- Mouse wheel → zoom in/out
- Click-drag → pan chart
- Double-click → reset zoom

**Animations:**
- Smooth transition when new predictions load (fade in)
- Progress bar during computation
- Pulse effect on "Update Now" when auto-refresh due

---

## Print/Export Layout

### PNG Export (1920x1080, 150 DPI)

```
┌───────────────────────────────────────────┐
│ Multi-Horizon Prediction Fan              │
│ BTCUSDT 1h | Generated: 2025-10-30 14:23  │
├───────────────────────────────────────────┤
│                                           │
│        [Prediction Fan Chart]             │
│          (Full width, 900px)              │
│                                           │
├───────────────────────────────────────────┤
│ Metrics Table (6 rows)                    │
├───────────────────────────────────────────┤
│ Model: artifacts/model_best.pt            │
│ Horizons: 1h, 3h, 5h, 10h, 20h, 30h      │
│ Generated by: BitCorn_Farmer Dashboard    │
└───────────────────────────────────────────┘
```

---

## Dark Theme (Planned)

### Color Adjustments

**Background:**
- Main: `#1e1e1e` (dark gray)
- Canvas: `#2d2d2d` (slightly lighter)
- Control panel: `#252526` (medium gray)

**Text:**
- Primary: `#ffffff` (white)
- Secondary: `#cccccc` (light gray)
- Labels: `#9d9d9d` (medium gray)

**Accents:**
- Success: `#4CAF50` (green)
- Error: `#f44336` (red)
- Warning: `#ff9800` (orange)
- Info: `#2196F3` (blue)

**Grid:**
- Color: `#404040` (dark gray)
- Alpha: 0.5

---

## Summary

The Prediction Dashboard features a **professional, clean layout** with:

✅ **Left Control Panel** (300px): All settings and controls
✅ **Center Visualization** (expandable): Large prediction fan chart
✅ **Bottom Metrics Table** (150px): Summary statistics
✅ **Responsive Design**: Adapts to window size
✅ **Professional Styling**: Trading platform aesthetic
✅ **Accessible Colors**: Multiple colormap options
✅ **Clear Visual Hierarchy**: Important info stands out
✅ **Intuitive Layout**: Similar to Bloomberg/TradingView

**Total Dimensions:**
- Minimum: 1000x700 (usable on laptops)
- Optimal: 1400x900 (desktop)
- Maximum: 1920x1080+ (large displays)

---

**End of Visual Layout Guide**
