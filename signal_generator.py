"""
Stock Signal Generator - Extracts intelligence from raw data for LLM consumption
Transforms database data into {symbol}_compiled.json
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional


class SignalGenerator:
    """Generate trading signals from stock database"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def _get_layer1_json(self, symbol: str, data_type: str) -> Optional[Dict]:
        """Fetch JSON data from layer1_data table"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT json_data FROM layer1_data WHERE symbol = ? AND data_type = ?',
            (symbol.upper(), data_type)
        )
        row = cursor.fetchone()
        if row and row['json_data']:
            try:
                return json.loads(row['json_data'])
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"  Warning: Failed to parse JSON for {symbol}/{data_type}: {e}")
                return None
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        formats = ["%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return None
    
    def extract_corporate_actions(self, symbol: str) -> Dict[str, Any]:
        """Extract and analyze corporate actions (bonus, splits, dividends, demergers)"""
        corp_info = self._get_layer1_json(symbol, "corporate_info")
        
        actions = {
            "bonuses": [],
            "splits": [],
            "dividends": [],
            "demergers": [],
            "others": [],
            "adjustment_factors": []
        }
        
        if not corp_info:
            return actions
        
        # Get corporate actions list
        corp_actions = corp_info.get("corporate_actions", {})
        if isinstance(corp_actions, dict):
            corp_actions = corp_actions.get("data", [])
        
        if not corp_actions:
            return actions
        
        for action in corp_actions:
            purpose = action.get("purpose", "").lower()
            ex_date = action.get("exdate", "")
            parsed_date = self._parse_date(ex_date)
            
            action_info = {
                "ex_date": ex_date,
                "parsed_date": parsed_date.isoformat() if parsed_date else None,
                "purpose": action.get("purpose", ""),
                "raw": action
            }
            
            # Classify and calculate adjustment factor
            if "bonus" in purpose:
                # Parse bonus ratio (e.g., "Bonus 1:1" = 1 new for 1 held = 2x adjustment)
                import re
                match = re.search(r'(\d+)\s*:\s*(\d+)', purpose)
                if match:
                    new_shares = int(match.group(1))
                    old_shares = int(match.group(2))
                    adj_factor = (old_shares + new_shares) / old_shares
                    action_info["ratio"] = f"{new_shares}:{old_shares}"
                    action_info["adjustment_factor"] = adj_factor
                    actions["adjustment_factors"].append({
                        "date": ex_date,
                        "type": "bonus",
                        "factor": adj_factor
                    })
                actions["bonuses"].append(action_info)
                
            elif "split" in purpose:
                # Parse split ratio (e.g., "Split 2:1" = 2 new for 1 old)
                import re
                match = re.search(r'(\d+)\s*:\s*(\d+)', purpose)
                if match:
                    new_shares = int(match.group(1))
                    old_shares = int(match.group(2))
                    adj_factor = new_shares / old_shares
                    action_info["ratio"] = f"{new_shares}:{old_shares}"
                    action_info["adjustment_factor"] = adj_factor
                    actions["adjustment_factors"].append({
                        "date": ex_date,
                        "type": "split",
                        "factor": adj_factor
                    })
                actions["splits"].append(action_info)
                
            elif "dividend" in purpose:
                # Extract dividend amount
                import re
                match = re.search(r'rs\.?\s*([\d.]+)', purpose)
                if match:
                    action_info["amount"] = float(match.group(1))
                actions["dividends"].append(action_info)
                
            elif "demerger" in purpose:
                actions["demergers"].append(action_info)
                # Demergers need manual adjustment - flag for review
                actions["adjustment_factors"].append({
                    "date": ex_date,
                    "type": "demerger",
                    "factor": None,  # Unknown without more data
                    "note": "Demerger requires manual price adjustment review"
                })
            else:
                actions["others"].append(action_info)
        
        return actions
    
    def get_adjusted_price_data(self, symbol: str) -> tuple:
        """Get price data with adjustment factors applied for corporate actions"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT trade_date, open_price, high_price, low_price, close_price, 
                   volume, delivery_qty, delivery_pct
            FROM price_volume_data 
            WHERE symbol = ? 
            ORDER BY trade_date ASC
        ''', (symbol.upper(),))
        
        rows = cursor.fetchall()
        if not rows:
            return [], {}
        
        # Get corporate actions
        corp_actions = self.extract_corporate_actions(symbol)
        adj_factors = corp_actions.get("adjustment_factors", [])
        
        # Convert adjustment dates
        adj_events = []
        for af in adj_factors:
            parsed = self._parse_date(af["date"])
            if parsed and af.get("factor"):
                adj_events.append({
                    "date": parsed,
                    "factor": af["factor"],
                    "type": af["type"]
                })
        
        # Sort by date descending (apply most recent first when going backwards)
        adj_events.sort(key=lambda x: x["date"], reverse=True)
        
        # Process rows and apply adjustments
        adjusted_data = []
        for row in rows:
            trade_date_str = row['trade_date']
            trade_date = self._parse_date(trade_date_str)
            
            # Calculate cumulative adjustment factor for dates before each event
            cumulative_factor = 1.0
            for event in adj_events:
                if trade_date and trade_date < event["date"]:
                    cumulative_factor *= event["factor"]
            
            adjusted_data.append({
                "trade_date": trade_date_str,
                "open": float(row['open_price'] or 0) / cumulative_factor,
                "high": float(row['high_price'] or 0) / cumulative_factor,
                "low": float(row['low_price'] or 0) / cumulative_factor,
                "close": float(row['close_price'] or 0) / cumulative_factor,
                "volume": float(row['volume'] or 0) * cumulative_factor,  # Volume increases with splits/bonus
                "delivery_qty": float(row['delivery_qty'] or 0),
                "delivery_pct": float(row['delivery_pct'] or 0),
                "adjustment_factor": cumulative_factor,
                "raw_close": float(row['close_price'] or 0)
            })
        
        return adjusted_data, corp_actions
    
    # ==================== INTRODUCTION EXTRACTION ====================
    
    def extract_introduction(self, symbol: str) -> Dict[str, Any]:
        """Extract equity_details, trade_info, corporate_info as introduction"""
        intro = {
            "symbol": symbol.upper(),
            "extracted_at": datetime.now().isoformat()
        }
        
        # Equity Details
        equity = self._get_layer1_json(symbol, "equity_details")
        if equity:
            info = equity.get("info", {})
            metadata = equity.get("metadata", {})
            security_info = equity.get("securityInfo", {})
            price_info = equity.get("priceInfo", {})
            
            intro["identity"] = {
                "company_name": info.get("companyName"),
                "symbol": info.get("symbol"),
                "isin": info.get("isin"),
                "series": info.get("activeSeries", []),
                "listing_date": info.get("listingDate"),
                "face_value": security_info.get("faceValue"),
                "issued_size": security_info.get("issuedSize")
            }
            
            intro["classification"] = {
                "industry": info.get("industry"),
                "sector": metadata.get("pdSectorInd"),
                "sector_indices": metadata.get("pdSectorIndAll", [])
            }
            
            intro["current_price"] = {
                "last_price": price_info.get("lastPrice"),
                "change": price_info.get("change"),
                "pct_change": price_info.get("pChange"),
                "open": price_info.get("open"),
                "high": price_info.get("intraDayHighLow", {}).get("max"),
                "low": price_info.get("intraDayHighLow", {}).get("min"),
                "close": price_info.get("close"),
                "vwap": price_info.get("vwap"),
                "week_high_52": price_info.get("weekHighLow", {}).get("max"),
                "week_low_52": price_info.get("weekHighLow", {}).get("min"),
                "upper_cp": price_info.get("upperCP"),
                "lower_cp": price_info.get("lowerCP")
            }
            
            intro["security_status"] = {
                "trading_status": security_info.get("tradingStatus"),
                "is_fno": security_info.get("isFNOSec", False),
                "is_slb": security_info.get("isSLBSec", False),
                "surveillance": security_info.get("surveillance", {})
            }
        
        # Trade Info
        trade = self._get_layer1_json(symbol, "trade_info")
        if trade:
            mkt_depth = trade.get("marketDeptOrderBook", {})
            # tradeInfo contains the actual volume/value data
            trade_info = mkt_depth.get("tradeInfo", {})
            sec_wise = trade.get("securityWiseDP", {})
            
            intro["trade_info"] = {
                "total_traded_volume": trade_info.get("totalTradedVolume"),
                "total_traded_value": trade_info.get("totalTradedValue"),
                "total_market_cap": trade_info.get("totalMarketCap"),
                "ffmc": trade_info.get("ffmc"),
                "impact_cost": trade_info.get("impactCost"),
                "cm_daily_volatility": trade_info.get("cmDailyVolatility"),
                "cm_annual_volatility": trade_info.get("cmAnnualVolatility"),
                "total_buy_qty": mkt_depth.get("totalBuyQuantity"),
                "total_sell_qty": mkt_depth.get("totalSellQuantity"),
                "delivery_qty": sec_wise.get("deliveryQuantity"),
                "delivery_pct": sec_wise.get("deliveryToTradedQuantity"),
                "cm_ffm_ratio": sec_wise.get("secWiseDelPosDate")
            }
        
        # Corporate Info
        corp = self._get_layer1_json(symbol, "corporate_info")
        if corp:
            # Extract announcements from latest_announcements.data
            announcements_raw = corp.get("latest_announcements", {})
            if isinstance(announcements_raw, dict):
                announcements = announcements_raw.get("data", [])[:5]
            else:
                announcements = announcements_raw[:5] if announcements_raw else []
            
            # Extract board meetings from borad_meeting.data (typo in DB key)
            board_raw = corp.get("borad_meeting", {})
            if isinstance(board_raw, dict):
                board_meetings = board_raw.get("data", [])[:3]
            else:
                board_meetings = board_raw[:3] if board_raw else []
            
            # Extract dividends from corporate_actions.data where purpose contains 'Dividend'
            actions_raw = corp.get("corporate_actions", {})
            if isinstance(actions_raw, dict):
                actions = actions_raw.get("data", [])
            else:
                actions = actions_raw if actions_raw else []
            
            dividends = [a for a in actions if "dividend" in a.get("purpose", "").lower()][:5]
            
            intro["corporate_info"] = {
                "announcements": announcements,
                "board_meetings": board_meetings,
                "dividends": dividends,
                "corporate_actions": actions[:5]  # Include all corporate actions
            }
        
        return intro
    
    # ==================== INTRADAY SIGNALS ====================
    
    def extract_intraday_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from intraday data"""
        intraday = self._get_layer1_json(symbol, "intraday_data")
        signals = {"available": False}
        
        if not intraday:
            return signals
        
        # Check for grapthData (actual key) or grappipts (alternative key)
        data_points = intraday.get("grapthData") or intraday.get("grappipts", [])
        if not data_points or len(data_points) < 2:
            return signals
        
        signals["available"] = True
        signals["close_price"] = intraday.get("closePrice")
        
        # Extract price series
        times = []
        prices = []
        for pt in data_points:
            if isinstance(pt, dict):
                times.append(pt.get("time", ""))
                prices.append(float(pt.get("ltp", pt.get("price", 0)) or 0))
            elif isinstance(pt, list) and len(pt) >= 2:
                # Format: [timestamp, price, 'PO'/'PC']
                times.append(pt[0])
                prices.append(float(pt[1] or 0))
        
        if len(prices) < 2:
            return signals
        
        prices = np.array(prices)
        valid_prices = prices[prices > 0]
        
        if len(valid_prices) < 2:
            return signals
        
        # 1. VWAP / Price Action
        current_price = valid_prices[-1]
        vwap = np.mean(valid_prices)  # Simplified VWAP
        twap = np.mean(valid_prices)  # Time-weighted (equal weights for simplicity)
        
        vwap_gap_pct = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
        
        signals["price_action"] = {
            "current_price": float(current_price),
            "vwap": float(vwap),
            "twap": float(twap),
            "vwap_gap_pct": round(vwap_gap_pct, 2),
            "vwap_signal": "Bullish" if vwap_gap_pct > 0.5 else ("Bearish" if vwap_gap_pct < -0.5 else "Neutral"),
            "intraday_high": float(np.max(valid_prices)),
            "intraday_low": float(np.min(valid_prices)),
            "opening_price": float(valid_prices[0]),
            "opening_range": float(np.max(valid_prices[:min(10, len(valid_prices))]) - np.min(valid_prices[:min(10, len(valid_prices))]))
        }
        
        # Trend detection (higher highs/lows)
        mid = len(valid_prices) // 2
        first_half_high = np.max(valid_prices[:mid]) if mid > 0 else valid_prices[0]
        second_half_high = np.max(valid_prices[mid:])
        first_half_low = np.min(valid_prices[:mid]) if mid > 0 else valid_prices[0]
        second_half_low = np.min(valid_prices[mid:])
        
        if second_half_high > first_half_high and second_half_low > first_half_low:
            trend = "Uptrend"
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        signals["price_action"]["intraday_trend"] = trend
        
        # 2. Volatility
        returns = np.diff(valid_prices) / valid_prices[:-1]
        realized_vol = float(np.std(returns) * np.sqrt(len(returns))) if len(returns) > 1 else 0
        
        # Volatility spikes (returns > 2 std)
        if len(returns) > 5:
            std_ret = np.std(returns)
            vol_spikes = int(np.sum(np.abs(returns) > 2 * std_ret))
        else:
            vol_spikes = 0
        
        price_range = float(np.max(valid_prices) - np.min(valid_prices))
        range_pct = (price_range / vwap) * 100 if vwap > 0 else 0
        
        signals["volatility"] = {
            "realized_intraday_vol": round(realized_vol * 100, 4),
            "volatility_spikes": vol_spikes,
            "range_expansion": round(range_pct, 2),
            "volatility_status": "High" if range_pct > 3 else ("Low" if range_pct < 1 else "Normal")
        }
        
        # 3. Momentum
        # Short-term momentum (price slope using linear regression)
        x = np.arange(len(valid_prices))
        slope = np.polyfit(x, valid_prices, 1)[0] if len(valid_prices) > 2 else 0
        momentum_score = (slope / vwap) * 100 if vwap > 0 else 0
        
        # Rate of Change
        roc_periods = min(20, len(valid_prices) - 1)
        roc = ((valid_prices[-1] - valid_prices[-1-roc_periods]) / valid_prices[-1-roc_periods]) * 100 if roc_periods > 0 and valid_prices[-1-roc_periods] > 0 else 0
        
        # Mean reversion signal
        price_vs_mean = (current_price - vwap) / vwap * 100 if vwap > 0 else 0
        mean_reversion = "Overbought" if price_vs_mean > 2 else ("Oversold" if price_vs_mean < -2 else "Neutral")
        
        signals["momentum"] = {
            "slope_momentum": round(momentum_score, 4),
            "rate_of_change": round(roc, 2),
            "mean_reversion_signal": mean_reversion,
            "momentum_bias": "Bullish" if momentum_score > 0 else "Bearish"
        }
        
        # 4. Market Microstructure
        tick_count = len(valid_prices)
        time_span_minutes = tick_count  # Assume 1-minute intervals
        tick_frequency = tick_count / max(time_span_minutes, 1)
        
        # Price stagnation (periods with < 0.1% move)
        small_moves = np.sum(np.abs(returns) < 0.001) if len(returns) > 0 else 0
        stagnation_pct = (small_moves / len(returns)) * 100 if len(returns) > 0 else 0
        
        signals["microstructure"] = {
            "tick_count": tick_count,
            "tick_frequency": round(tick_frequency, 2),
            "stagnation_pct": round(stagnation_pct, 2),
            "activity_level": "High" if tick_frequency > 1 else "Normal"
        }
        
        # 5. Session Insights
        n = len(valid_prices)
        open_third = valid_prices[:n//3] if n >= 3 else valid_prices
        mid_third = valid_prices[n//3:2*n//3] if n >= 3 else valid_prices
        close_third = valid_prices[2*n//3:] if n >= 3 else valid_prices
        
        open_avg = float(np.mean(open_third))
        mid_avg = float(np.mean(mid_third))
        close_avg = float(np.mean(close_third))
        
        if close_avg > mid_avg > open_avg:
            session_pattern = "Strong Trend Day"
        elif close_avg < mid_avg < open_avg:
            session_pattern = "Weak Trend Day"
        elif abs(close_avg - open_avg) < (vwap * 0.005):
            session_pattern = "Range Bound"
        else:
            session_pattern = "Mixed"
        
        signals["session_insights"] = {
            "open_session_avg": round(open_avg, 2),
            "mid_session_avg": round(mid_avg, 2),
            "close_session_avg": round(close_avg, 2),
            "session_pattern": session_pattern,
            "trend_continuation": close_avg > open_avg
        }
        
        # 6. Strategy Inputs
        atr_proxy = float(np.mean(np.abs(np.diff(valid_prices)))) if len(valid_prices) > 1 else 0
        
        signals["strategy_inputs"] = {
            "scalp_range": round(atr_proxy, 2),
            "stop_loss_suggestion": round(atr_proxy * 2, 2),
            "position_sizing_vol": round(realized_vol, 4),
            "trend_following_bias": trend
        }
        
        return signals
    
    # ==================== HISTORICAL SIGNALS ====================
    
    def extract_historical_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from historical data with corporate action adjustments"""
        signals = {"available": False}
        
        # Get adjusted price data (accounts for bonuses, splits, etc.)
        adjusted_data, corp_actions = self.get_adjusted_price_data(symbol)
        
        if not adjusted_data or len(adjusted_data) < 20:
            return signals
        
        signals["available"] = True
        
        # Add corporate actions summary
        signals["corporate_actions"] = {
            "bonuses": [{"date": b["ex_date"], "ratio": b.get("ratio"), "adj_factor": b.get("adjustment_factor")} for b in corp_actions.get("bonuses", [])],
            "splits": [{"date": s["ex_date"], "ratio": s.get("ratio"), "adj_factor": s.get("adjustment_factor")} for s in corp_actions.get("splits", [])],
            "demergers": [{"date": d["ex_date"], "purpose": d.get("purpose")} for d in corp_actions.get("demergers", [])],
            "recent_dividends": [{"date": d["ex_date"], "amount": d.get("amount")} for d in corp_actions.get("dividends", [])[:5]],
            "price_adjusted": len(corp_actions.get("adjustment_factors", [])) > 0,
            "adjustment_note": "Historical prices adjusted for corporate actions (bonus/splits). Volume adjusted inversely."
        }
        
        # Convert to arrays - use ADJUSTED prices for analysis
        dates = [r['trade_date'] for r in adjusted_data]
        opens = np.array([r['open'] for r in adjusted_data])
        highs = np.array([r['high'] for r in adjusted_data])
        lows = np.array([r['low'] for r in adjusted_data])
        closes = np.array([r['close'] for r in adjusted_data])
        volumes = np.array([r['volume'] for r in adjusted_data])
        delivery_pct = np.array([r['delivery_pct'] for r in adjusted_data])
        
        # Also track raw (unadjusted) current price for reference
        raw_current = adjusted_data[-1]['raw_close']
        signals["current_raw_price"] = raw_current
        
        current_close = closes[-1]
        
        # 1. Trend & Structure
        # SMA calculations
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)
        sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else np.mean(closes)
        
        # Trend determination
        if current_close > sma_50 > sma_200:
            trend = "Strong Uptrend"
        elif current_close > sma_200:
            trend = "Uptrend"
        elif current_close < sma_50 < sma_200:
            trend = "Strong Downtrend"
        elif current_close < sma_200:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        # Higher highs / lower lows (last 20 days)
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
        
        structure = "Bullish" if hh_count > ll_count else ("Bearish" if ll_count > hh_count else "Neutral")
        
        # 52-week high/low
        week_52_high = float(np.max(highs[-252:])) if len(highs) >= 252 else float(np.max(highs))
        week_52_low = float(np.min(lows[-252:])) if len(lows) >= 252 else float(np.min(lows))
        near_52_high = (week_52_high - current_close) / week_52_high * 100 if week_52_high > 0 else 0
        
        signals["trend_structure"] = {
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
            "current_vs_sma200_pct": round((current_close - sma_200) / sma_200 * 100, 2) if sma_200 > 0 else 0,
            "trend": trend,
            "price_structure": structure,
            "higher_highs_count_20d": hh_count,
            "lower_lows_count_20d": ll_count,
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
            "pct_from_52_high": round(near_52_high, 2),
            "near_ath": near_52_high < 5
        }
        
        # 2. Returns & Volatility
        returns = np.diff(closes) / closes[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        daily_return = float(returns[-1]) * 100 if len(returns) > 0 else 0
        weekly_return = float(np.sum(returns[-5:])) * 100 if len(returns) >= 5 else 0
        monthly_return = float(np.sum(returns[-22:])) * 100 if len(returns) >= 22 else 0
        yearly_return = float(np.sum(returns[-252:])) * 100 if len(returns) >= 252 else float(np.sum(returns)) * 100
        
        # Volatility
        historical_vol = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else 0
        
        # ATR (Average True Range)
        tr = np.maximum(highs[1:] - lows[1:], 
                       np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                 np.abs(lows[1:] - closes[:-1])))
        atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr)) if len(tr) > 0 else 0
        
        # Max drawdown
        cummax = np.maximum.accumulate(closes)
        drawdowns = (cummax - closes) / cummax
        max_drawdown = float(np.max(drawdowns)) * 100
        
        signals["returns_volatility"] = {
            "daily_return_pct": round(daily_return, 2),
            "weekly_return_pct": round(weekly_return, 2),
            "monthly_return_pct": round(monthly_return, 2),
            "yearly_return_pct": round(yearly_return, 2),
            "annualized_volatility": round(historical_vol * 100, 2),
            "atr_14": round(atr_14, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "risk_level": "High" if historical_vol > 0.4 else ("Low" if historical_vol < 0.2 else "Medium")
        }
        
        # 3. Volume & Participation
        avg_volume_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        current_volume = float(volumes[-1])
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # Volume spikes (> 2x average)
        volume_spikes = int(np.sum(volumes[-20:] > 2 * avg_volume_20))
        
        # Delivery analysis
        avg_delivery = float(np.mean(delivery_pct[-20:])) if len(delivery_pct) >= 20 else float(np.mean(delivery_pct))
        current_delivery = float(delivery_pct[-1])
        
        # Price-volume divergence
        price_up = closes[-1] > closes[-2] if len(closes) >= 2 else False
        volume_up = volumes[-1] > volumes[-2] if len(volumes) >= 2 else False
        
        if price_up and volume_up:
            pv_signal = "Bullish Confirmation"
        elif price_up and not volume_up:
            pv_signal = "Weak Rally (Low Volume)"
        elif not price_up and volume_up:
            pv_signal = "Selling Pressure"
        else:
            pv_signal = "Low Activity"
        
        signals["volume_participation"] = {
            "current_volume": current_volume,
            "avg_volume_20d": round(avg_volume_20, 0),
            "volume_ratio": round(volume_ratio, 2),
            "volume_spikes_20d": volume_spikes,
            "avg_delivery_pct": round(avg_delivery, 2),
            "current_delivery_pct": round(current_delivery, 2),
            "price_volume_signal": pv_signal,
            "accumulation_signal": "Accumulation" if current_delivery > avg_delivery and price_up else ("Distribution" if current_delivery < avg_delivery and not price_up else "Neutral")
        }
        
        # 4. Market Strength
        # Close vs VWAP proxy (using typical price)
        typical_price = (highs + lows + closes) / 3
        vwap_proxy = float(np.sum(typical_price[-20:] * volumes[-20:]) / np.sum(volumes[-20:])) if np.sum(volumes[-20:]) > 0 else current_close
        
        close_vs_vwap = "Bullish" if current_close > vwap_proxy else "Bearish"
        
        # Open-Close behavior
        bullish_days = int(np.sum(closes[-20:] > opens[-20:]))
        bearish_days = 20 - bullish_days
        
        signals["market_strength"] = {
            "vwap_20d": round(vwap_proxy, 2),
            "close_vs_vwap": close_vs_vwap,
            "bullish_days_20": bullish_days,
            "bearish_days_20": bearish_days,
            "market_control": "Bulls" if bullish_days > 12 else ("Bears" if bearish_days > 12 else "Contested")
        }
        
        # 5. Key Levels (Support & Resistance)
        # Use pivot points and volume-weighted levels
        pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
        r1 = 2 * pivot - lows[-1]
        r2 = pivot + (highs[-1] - lows[-1])
        s1 = 2 * pivot - highs[-1]
        s2 = pivot - (highs[-1] - lows[-1])
        
        signals["key_levels"] = {
            "pivot": round(pivot, 2),
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "week_52_high": week_52_high,
            "week_52_low": week_52_low
        }
        
        # 6. Strategy Signals
        # RSI calculation
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        signals["strategy_signals"] = {
            "rsi_14": round(rsi, 2),
            "rsi_signal": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
            "stop_loss_atr": round(current_close - 2 * atr_14, 2),
            "target_atr": round(current_close + 3 * atr_14, 2),
            "swing_bias": trend
        }
        
        return signals
    
    # ==================== ADVANCED PRICE-VOLUME SIGNALS ====================
    
    def extract_advanced_price_volume_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract comprehensive signals from price_volume_data table:
        - Price Action: Trend, Support/Resistance, Breakouts, ATR
        - Volume Intelligence: Expansion/contraction, Climax volume
        - Delivery-Based Smart Money: Accumulation/Distribution
        - Technical Indicators: MACD, EMA, OBV, A/D Line
        """
        signals = {"available": False}
        
        # Fetch data directly from price_volume_data table
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT trade_date, open_price, high_price, low_price, close_price, 
                   volume, delivery_qty, delivery_pct
            FROM price_volume_data 
            WHERE symbol = ?
            ORDER BY id DESC
        ''', (symbol.upper(),))
        
        rows = cursor.fetchall()
        if len(rows) < 50:
            return signals
        
        # Reverse to chronological order (oldest first)
        rows = list(reversed(rows))
        
        # Extract arrays
        dates = [r[0] for r in rows]
        opens = np.array([float(r[1] or 0) for r in rows])
        highs = np.array([float(r[2] or 0) for r in rows])
        lows = np.array([float(r[3] or 0) for r in rows])
        closes = np.array([float(r[4] or 0) for r in rows])
        volumes = np.array([float(r[5] or 0) for r in rows])
        delivery_qty = np.array([float(r[6] or 0) for r in rows])
        delivery_pct = np.array([float(r[7] or 0) for r in rows])
        
        signals["available"] = True
        signals["data_points"] = len(rows)
        signals["date_range"] = {"from": dates[0], "to": dates[-1]}
        
        # ========== 1. PRICE ACTION SIGNALS ==========
        
        # EMA Calculations (more responsive than SMA)
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        ema_12 = ema(closes, 12)
        ema_26 = ema(closes, 26)
        ema_50 = ema(closes, 50)
        ema_200 = ema(closes, 200)
        
        # Current values
        current_close = closes[-1]
        current_ema12 = ema_12[-1]
        current_ema26 = ema_26[-1]
        current_ema50 = ema_50[-1]
        current_ema200 = ema_200[-1]
        
        # Trend Analysis - Higher Highs / Lower Lows (20-day swing analysis)
        swing_period = 20
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_period, len(highs) - swing_period):
            if highs[i] == max(highs[i-swing_period//2:i+swing_period//2+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-swing_period//2:i+swing_period//2+1]):
                swing_lows.append((i, lows[i]))
        
        # Check for higher highs and higher lows (uptrend) or lower highs and lower lows (downtrend)
        recent_swing_highs = [h[1] for h in swing_highs[-4:]] if len(swing_highs) >= 4 else []
        recent_swing_lows = [l[1] for l in swing_lows[-4:]] if len(swing_lows) >= 4 else []
        
        hh_pattern = all(recent_swing_highs[i] > recent_swing_highs[i-1] for i in range(1, len(recent_swing_highs))) if len(recent_swing_highs) >= 2 else False
        hl_pattern = all(recent_swing_lows[i] > recent_swing_lows[i-1] for i in range(1, len(recent_swing_lows))) if len(recent_swing_lows) >= 2 else False
        lh_pattern = all(recent_swing_highs[i] < recent_swing_highs[i-1] for i in range(1, len(recent_swing_highs))) if len(recent_swing_highs) >= 2 else False
        ll_pattern = all(recent_swing_lows[i] < recent_swing_lows[i-1] for i in range(1, len(recent_swing_lows))) if len(recent_swing_lows) >= 2 else False
        
        if hh_pattern and hl_pattern:
            price_structure = "Strong Uptrend (HH + HL)"
        elif hh_pattern or hl_pattern:
            price_structure = "Uptrend"
        elif lh_pattern and ll_pattern:
            price_structure = "Strong Downtrend (LH + LL)"
        elif lh_pattern or ll_pattern:
            price_structure = "Downtrend"
        else:
            price_structure = "Consolidation"
        
        # Support & Resistance using volume-weighted price levels
        # Find high-volume price zones
        price_bins = 50
        price_range = np.linspace(min(lows), max(highs), price_bins)
        volume_profile = np.zeros(price_bins - 1)
        
        for i in range(len(closes)):
            for j in range(len(price_range) - 1):
                if price_range[j] <= closes[i] < price_range[j + 1]:
                    volume_profile[j] += volumes[i]
                    break
        
        # Top volume nodes = potential support/resistance
        top_nodes_idx = np.argsort(volume_profile)[-3:]
        key_levels = sorted([round((price_range[i] + price_range[i+1]) / 2, 2) for i in top_nodes_idx])
        
        # Classify as support or resistance relative to current price
        support_levels = [l for l in key_levels if l < current_close]
        resistance_levels = [l for l in key_levels if l > current_close]
        
        # Breakout / Breakdown Detection
        recent_high_20 = max(highs[-20:])
        recent_low_20 = min(lows[-20:])
        recent_high_50 = max(highs[-50:])
        recent_low_50 = min(lows[-50:])
        
        breakout_signal = None
        if current_close > recent_high_20 and volumes[-1] > np.mean(volumes[-20:]) * 1.5:
            breakout_signal = "20-Day Breakout (Volume Confirmed)"
        elif current_close > recent_high_50 and volumes[-1] > np.mean(volumes[-50:]) * 1.5:
            breakout_signal = "50-Day Breakout (Volume Confirmed)"
        elif current_close < recent_low_20 and volumes[-1] > np.mean(volumes[-20:]) * 1.5:
            breakout_signal = "20-Day Breakdown (Volume Confirmed)"
        elif current_close < recent_low_50 and volumes[-1] > np.mean(volumes[-50:]) * 1.5:
            breakout_signal = "50-Day Breakdown (Volume Confirmed)"
        elif current_close > recent_high_20:
            breakout_signal = "20-Day Breakout (Low Volume - Weak)"
        elif current_close < recent_low_20:
            breakout_signal = "20-Day Breakdown (Low Volume - Weak)"
        else:
            breakout_signal = "No Breakout"
        
        # ATR & Volatility
        tr = np.maximum(highs[1:] - lows[1:], 
                       np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                 np.abs(lows[1:] - closes[:-1])))
        atr_14 = np.mean(tr[-14:])
        atr_50 = np.mean(tr[-50:])
        volatility_expansion = atr_14 > atr_50 * 1.2
        
        signals["price_action"] = {
            "current_price": round(current_close, 2),
            "ema_12": round(current_ema12, 2),
            "ema_26": round(current_ema26, 2),
            "ema_50": round(current_ema50, 2),
            "ema_200": round(current_ema200, 2),
            "ema_trend": "Bullish" if current_ema12 > current_ema26 > current_ema50 else ("Bearish" if current_ema12 < current_ema26 < current_ema50 else "Mixed"),
            "price_vs_ema200": "Above" if current_close > current_ema200 else "Below",
            "price_structure": price_structure,
            "swing_highs_recent": [round(h, 2) for h in recent_swing_highs[-3:]],
            "swing_lows_recent": [round(l, 2) for l in recent_swing_lows[-3:]],
            "volume_based_support": support_levels,
            "volume_based_resistance": resistance_levels,
            "breakout_signal": breakout_signal,
            "atr_14": round(atr_14, 2),
            "atr_50": round(atr_50, 2),
            "volatility_state": "Expanding" if volatility_expansion else "Contracting",
            "atr_stop_loss": round(current_close - 2 * atr_14, 2),
            "atr_target": round(current_close + 3 * atr_14, 2)
        }
        
        # ========== 2. VOLUME INTELLIGENCE ==========
        
        avg_vol_5 = np.mean(volumes[-5:])
        avg_vol_20 = np.mean(volumes[-20:])
        avg_vol_50 = np.mean(volumes[-50:])
        current_vol = volumes[-1]
        
        # Volume expansion/contraction
        vol_ratio_20 = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1
        vol_trend = "Expansion" if avg_vol_5 > avg_vol_20 * 1.2 else ("Contraction" if avg_vol_5 < avg_vol_20 * 0.8 else "Normal")
        
        # Price-Volume Confirmation
        price_change_5d = (closes[-1] - closes[-6]) / closes[-6] * 100 if closes[-6] > 0 else 0
        vol_change_5d = (avg_vol_5 - avg_vol_20) / avg_vol_20 * 100 if avg_vol_20 > 0 else 0
        
        if price_change_5d > 0 and vol_change_5d > 0:
            pv_confirmation = "Bullish Confirmation (Price Up + Volume Up)"
        elif price_change_5d < 0 and vol_change_5d > 0:
            pv_confirmation = "Bearish Confirmation (Price Down + Volume Up)"
        elif price_change_5d > 0 and vol_change_5d < 0:
            pv_confirmation = "Divergence Warning (Price Up + Volume Down)"
        elif price_change_5d < 0 and vol_change_5d < 0:
            pv_confirmation = "Weak Selling (Price Down + Volume Down)"
        else:
            pv_confirmation = "Neutral"
        
        # Climax Volume Detection (potential reversal signal)
        vol_std = np.std(volumes[-50:])
        vol_zscore = (current_vol - avg_vol_50) / vol_std if vol_std > 0 else 0
        
        climax_signal = None
        if vol_zscore > 2.5:
            if closes[-1] > opens[-1]:
                climax_signal = "Buying Climax (Potential Top)"
            else:
                climax_signal = "Selling Climax (Potential Bottom)"
        elif vol_zscore > 2:
            climax_signal = "High Volume Alert"
        else:
            climax_signal = "Normal Volume"
        
        # OBV (On-Balance Volume)
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        obv_trend = "Rising" if obv[-1] > obv[-20] else "Falling"
        obv_divergence = None
        if closes[-1] > closes[-20] and obv[-1] < obv[-20]:
            obv_divergence = "Bearish Divergence (Price Up, OBV Down)"
        elif closes[-1] < closes[-20] and obv[-1] > obv[-20]:
            obv_divergence = "Bullish Divergence (Price Down, OBV Up)"
        
        signals["volume_intelligence"] = {
            "current_volume": int(current_vol),
            "avg_volume_5d": int(avg_vol_5),
            "avg_volume_20d": int(avg_vol_20),
            "avg_volume_50d": int(avg_vol_50),
            "volume_ratio": round(vol_ratio_20, 2),
            "volume_trend": vol_trend,
            "price_change_5d_pct": round(price_change_5d, 2),
            "volume_change_5d_pct": round(vol_change_5d, 2),
            "price_volume_confirmation": pv_confirmation,
            "volume_zscore": round(vol_zscore, 2),
            "climax_signal": climax_signal,
            "obv_current": int(obv[-1]),
            "obv_20d_ago": int(obv[-20]),
            "obv_trend": obv_trend,
            "obv_divergence": obv_divergence
        }
        
        # ========== 3. DELIVERY-BASED SMART MONEY SIGNALS ==========
        
        avg_delivery_20 = np.mean(delivery_pct[-20:])
        avg_delivery_50 = np.mean(delivery_pct[-50:])
        current_delivery = delivery_pct[-1]
        
        # Smart Money Detection
        high_delivery_threshold = avg_delivery_50 * 1.15  # 15% above average
        low_delivery_threshold = avg_delivery_50 * 0.85  # 15% below average
        
        price_up = closes[-1] > closes[-2]
        high_delivery = current_delivery > high_delivery_threshold
        low_delivery = current_delivery < low_delivery_threshold
        high_volume = current_vol > avg_vol_20 * 1.3
        
        if high_delivery and price_up:
            smart_money_signal = "Strong Accumulation (High Delivery + Price Up)"
        elif high_delivery and not price_up:
            smart_money_signal = "Distribution Warning (High Delivery + Price Down)"
        elif low_delivery and high_volume:
            smart_money_signal = "Speculation/Intraday Churn (Low Delivery + High Volume)"
        elif low_delivery and not price_up:
            smart_money_signal = "Weak Hands Selling (Low Delivery + Price Down)"
        else:
            smart_money_signal = "Neutral Activity"
        
        # Accumulation-Distribution Line
        clv = np.zeros(len(closes))  # Close Location Value
        for i in range(len(closes)):
            hl_range = highs[i] - lows[i]
            if hl_range > 0:
                clv[i] = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl_range
            else:
                clv[i] = 0
        
        ad_line = np.cumsum(clv * volumes)
        ad_trend = "Accumulation" if ad_line[-1] > ad_line[-20] else "Distribution"
        
        # Delivery trend over time
        delivery_trend_20 = "Increasing" if np.mean(delivery_pct[-5:]) > np.mean(delivery_pct[-20:-5]) else "Decreasing"
        
        signals["smart_money_signals"] = {
            "current_delivery_pct": round(current_delivery, 2),
            "avg_delivery_20d": round(avg_delivery_20, 2),
            "avg_delivery_50d": round(avg_delivery_50, 2),
            "delivery_vs_avg": "High" if high_delivery else ("Low" if low_delivery else "Normal"),
            "delivery_trend": delivery_trend_20,
            "smart_money_signal": smart_money_signal,
            "ad_line_current": round(ad_line[-1], 0),
            "ad_line_20d_ago": round(ad_line[-20], 0),
            "ad_trend": ad_trend,
            "institutional_activity": "Active" if high_delivery and high_volume else "Normal"
        }
        
        # ========== 4. TECHNICAL INDICATORS ==========
        
        # MACD
        macd_line = ema_12 - ema_26
        macd_signal = ema(macd_line, 9)
        macd_histogram = macd_line - macd_signal
        
        macd_crossover = None
        if macd_line[-1] > macd_signal[-1] and macd_line[-2] <= macd_signal[-2]:
            macd_crossover = "Bullish Crossover"
        elif macd_line[-1] < macd_signal[-1] and macd_line[-2] >= macd_signal[-2]:
            macd_crossover = "Bearish Crossover"
        
        macd_momentum = "Bullish" if macd_histogram[-1] > 0 else "Bearish"
        macd_strength = "Strengthening" if abs(macd_histogram[-1]) > abs(macd_histogram[-2]) else "Weakening"
        
        # RSI with divergence detection
        returns = np.diff(closes) / closes[:-1]
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        # Calculate RSI for last 100 days for divergence analysis
        rsi_values = []
        for i in range(14, len(returns)):
            avg_gain = np.mean(gains[i-14:i])
            avg_loss = np.mean(losses[i-14:i])
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi_values.append(100 - (100 / (1 + rs)))
        
        current_rsi = rsi_values[-1] if rsi_values else 50
        
        # RSI Divergence
        rsi_divergence = None
        if len(rsi_values) >= 20:
            if closes[-1] > closes[-20] and rsi_values[-1] < rsi_values[-20]:
                rsi_divergence = "Bearish Divergence"
            elif closes[-1] < closes[-20] and rsi_values[-1] > rsi_values[-20]:
                rsi_divergence = "Bullish Divergence"
        
        signals["technical_indicators"] = {
            "macd_line": round(macd_line[-1], 2),
            "macd_signal": round(macd_signal[-1], 2),
            "macd_histogram": round(macd_histogram[-1], 2),
            "macd_crossover": macd_crossover,
            "macd_momentum": macd_momentum,
            "macd_strength": macd_strength,
            "rsi_14": round(current_rsi, 2),
            "rsi_signal": "Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral"),
            "rsi_divergence": rsi_divergence,
            "rsi_20d_ago": round(rsi_values[-20], 2) if len(rsi_values) >= 20 else None
        }
        
        # ========== 5. COMPOSITE SIGNALS SUMMARY ==========
        
        bullish_signals = 0
        bearish_signals = 0
        
        # Count bullish signals
        if "Uptrend" in price_structure: bullish_signals += 1
        if current_close > current_ema200: bullish_signals += 1
        if "Breakout" in breakout_signal and "Breakdown" not in breakout_signal: bullish_signals += 1
        if "Bullish" in pv_confirmation: bullish_signals += 1
        if "Accumulation" in smart_money_signal: bullish_signals += 1
        if ad_trend == "Accumulation": bullish_signals += 1
        if macd_momentum == "Bullish": bullish_signals += 1
        if current_rsi > 50 and current_rsi < 70: bullish_signals += 1
        if obv_trend == "Rising": bullish_signals += 1
        
        # Count bearish signals
        if "Downtrend" in price_structure: bearish_signals += 1
        if current_close < current_ema200: bearish_signals += 1
        if "Breakdown" in breakout_signal: bearish_signals += 1
        if "Bearish" in pv_confirmation: bearish_signals += 1
        if "Distribution" in smart_money_signal: bearish_signals += 1
        if ad_trend == "Distribution": bearish_signals += 1
        if macd_momentum == "Bearish": bearish_signals += 1
        if current_rsi > 70: bearish_signals += 1
        if obv_trend == "Falling": bearish_signals += 1
        
        total_signals = bullish_signals + bearish_signals
        
        if bullish_signals > bearish_signals + 2:
            composite_bias = "Strong Bullish"
        elif bullish_signals > bearish_signals:
            composite_bias = "Bullish"
        elif bearish_signals > bullish_signals + 2:
            composite_bias = "Strong Bearish"
        elif bearish_signals > bullish_signals:
            composite_bias = "Bearish"
        else:
            composite_bias = "Neutral"
        
        signals["composite_summary"] = {
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "signal_ratio": f"{bullish_signals}:{bearish_signals}",
            "composite_bias": composite_bias,
            "confidence": "High" if abs(bullish_signals - bearish_signals) >= 3 else "Medium" if abs(bullish_signals - bearish_signals) >= 1 else "Low"
        }
        
        return signals
    
    # ==================== OPTION CHAIN SIGNALS ====================
    
    def _parse_expiry_date(self, date_str: str) -> Optional[datetime]:
        """Parse expiry date string to datetime"""
        formats = ["%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return None
    
    def _analyze_single_expiry(self, expiry_data: List, current_price: float) -> Dict:
        """Analyze option chain for a single expiry"""
        # Separate calls and puts
        calls = {r[1]: {"oi": r[3] or 0, "chg_oi": r[4] or 0, "vol": r[5] or 0, 
                        "iv": r[6] or 0, "ltp": r[7] or 0} for r in expiry_data if r[2] == 'CE'}
        puts = {r[1]: {"oi": r[3] or 0, "chg_oi": r[4] or 0, "vol": r[5] or 0, 
                       "iv": r[6] or 0, "ltp": r[7] or 0} for r in expiry_data if r[2] == 'PE'}
        
        all_strikes = sorted(set(list(calls.keys()) + list(puts.keys())))
        if not all_strikes:
            return None
        
        # PCR calculation
        total_call_oi = sum(c["oi"] for c in calls.values())
        total_put_oi = sum(p["oi"] for p in puts.values())
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Max Pain calculation
        max_pain_strike = None
        min_pain = float('inf')
        for strike in all_strikes:
            pain = 0
            for s in all_strikes:
                if s in calls and s < strike:
                    pain += calls[s]["oi"] * (strike - s)
                if s in puts and s > strike:
                    pain += puts[s]["oi"] * (s - strike)
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike
        
        # OI buildup
        put_oi_buildup = sum(p["chg_oi"] for p in puts.values() if p["chg_oi"] > 0)
        call_oi_buildup = sum(c["chg_oi"] for c in calls.values() if c["chg_oi"] > 0)
        
        # Top OI levels
        top_put_oi = sorted([(s, puts[s]["oi"]) for s in puts], key=lambda x: x[1], reverse=True)[:3]
        top_call_oi = sorted([(s, calls[s]["oi"]) for s in calls], key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "pcr_oi": pcr_oi,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "max_pain": max_pain_strike,
            "put_oi_buildup": put_oi_buildup,
            "call_oi_buildup": call_oi_buildup,
            "top_put_support": top_put_oi[0][0] if top_put_oi else None,
            "top_call_resistance": top_call_oi[0][0] if top_call_oi else None,
            "calls": calls,
            "puts": puts,
            "all_strikes": all_strikes
        }
    
    def extract_option_chain_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract signals from option_chain_data table:
        - Market Expectation: PCR, Max Pain, ATM/ITM/OTM activity
        - Smart Money Positioning: OI buildup patterns
        - Options-based Support/Resistance: Highest OI levels
        - Volatility & Risk: IV, IV skew, Expected move
        - Multi-expiry analysis for trend confirmation
        """
        signals = {"available": False}
        
        # Fetch option chain data
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT expiry_date, strike_price, option_type, open_interest, 
                   change_in_oi, volume, iv, ltp, bid_price, ask_price
            FROM option_chain_data 
            WHERE symbol = ?
            ORDER BY expiry_date, strike_price
        ''', (symbol.upper(),))
        
        rows = cursor.fetchall()
        if not rows or len(rows) < 10:
            return signals
        
        # Get current stock price
        cursor.execute('''
            SELECT close_price FROM price_volume_data 
            WHERE symbol = ? ORDER BY id DESC LIMIT 1
        ''', (symbol.upper(),))
        price_row = cursor.fetchone()
        
        # Find ATM strike from option chain
        all_strikes = sorted(set(r[1] for r in rows if r[3] > 0))
        if not all_strikes:
            return signals
        
        mid_strike = all_strikes[len(all_strikes) // 2] if all_strikes else 1500
        current_price = mid_strike
        
        if price_row and price_row[0]:
            raw_price = price_row[0]
            if min(all_strikes) <= raw_price <= max(all_strikes):
                current_price = raw_price
        
        signals["available"] = True
        signals["reference_price"] = round(current_price, 2)
        
        # Sort expiries by date (nearest first)
        expiry_strings = list(set(r[0] for r in rows))
        today = datetime.now()
        
        expiry_dates = []
        for exp_str in expiry_strings:
            exp_dt = self._parse_expiry_date(exp_str)
            if exp_dt:
                days_to_expiry = (exp_dt - today).days
                expiry_dates.append((exp_str, exp_dt, days_to_expiry))
        
        # Sort by days to expiry (nearest first)
        expiry_dates.sort(key=lambda x: x[2])
        
        if not expiry_dates:
            return signals
        
        signals["expiries_available"] = [
            {"expiry": e[0], "days_to_expiry": e[2]} for e in expiry_dates
        ]
        
        # ========== MULTI-EXPIRY ANALYSIS ==========
        multi_expiry_analysis = {}
        
        for exp_str, exp_dt, days in expiry_dates:
            expiry_data = [r for r in rows if r[0] == exp_str]
            analysis = self._analyze_single_expiry(expiry_data, current_price)
            if analysis:
                multi_expiry_analysis[exp_str] = {
                    "days_to_expiry": days,
                    "pcr_oi": round(analysis["pcr_oi"], 2),
                    "max_pain": analysis["max_pain"],
                    "put_support": analysis["top_put_support"],
                    "call_resistance": analysis["top_call_resistance"],
                    "put_oi_buildup": int(analysis["put_oi_buildup"]),
                    "call_oi_buildup": int(analysis["call_oi_buildup"]),
                    "total_call_oi": int(analysis["total_call_oi"]),
                    "total_put_oi": int(analysis["total_put_oi"])
                }
        
        signals["multi_expiry_analysis"] = multi_expiry_analysis
        
        # ========== PRIMARY ANALYSIS: NEAREST EXPIRY ==========
        nearest_expiry = expiry_dates[0][0]
        nearest_days = expiry_dates[0][2]
        signals["primary_expiry"] = nearest_expiry
        signals["days_to_expiry"] = nearest_days
        
        expiry_data = [r for r in rows if r[0] == nearest_expiry]
        
        # Separate calls and puts for nearest expiry
        calls = {r[1]: {"oi": r[3] or 0, "chg_oi": r[4] or 0, "vol": r[5] or 0, 
                        "iv": r[6] or 0, "ltp": r[7] or 0} for r in expiry_data if r[2] == 'CE'}
        puts = {r[1]: {"oi": r[3] or 0, "chg_oi": r[4] or 0, "vol": r[5] or 0, 
                       "iv": r[6] or 0, "ltp": r[7] or 0} for r in expiry_data if r[2] == 'PE'}
        
        all_strikes = sorted(set(list(calls.keys()) + list(puts.keys())))
        
        if not all_strikes:
            signals["available"] = False
            return signals
        
        # ========== 1. MARKET EXPECTATION & BIAS ==========
        
        # Put-Call Ratio (PCR)
        total_call_oi = sum(c["oi"] for c in calls.values())
        total_put_oi = sum(p["oi"] for p in puts.values())
        total_call_vol = sum(c["vol"] for c in calls.values())
        total_put_vol = sum(p["vol"] for p in puts.values())
        
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # PCR interpretation
        if pcr_oi > 1.2:
            pcr_signal = "Bullish (High Put Writing)"
        elif pcr_oi > 0.8:
            pcr_signal = "Neutral"
        else:
            pcr_signal = "Bearish (Low Put Support)"
        
        # Max Pain Calculation
        # Max pain = strike where option buyers lose the most (writers gain the most)
        max_pain_strike = None
        min_pain = float('inf')
        
        for strike in all_strikes:
            pain = 0
            # For each strike, calculate total pain if stock expires at this level
            for s in all_strikes:
                if s in calls:
                    # Call buyers lose if strike < expiry_price (ITM calls expire worthless)
                    if s < strike:
                        pain += calls[s]["oi"] * (strike - s)
                if s in puts:
                    # Put buyers lose if strike > expiry_price (ITM puts expire worthless)
                    if s > strike:
                        pain += puts[s]["oi"] * (s - strike)
            
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike
        
        # ATM / ITM / OTM Activity
        atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))
        atm_range = [s for s in all_strikes if abs(s - current_price) <= current_price * 0.03]
        itm_calls = [s for s in calls.keys() if s < current_price]
        otm_calls = [s for s in calls.keys() if s > current_price]
        itm_puts = [s for s in puts.keys() if s > current_price]
        otm_puts = [s for s in puts.keys() if s < current_price]
        
        atm_call_oi = sum(calls.get(s, {}).get("oi", 0) for s in atm_range)
        atm_put_oi = sum(puts.get(s, {}).get("oi", 0) for s in atm_range)
        otm_call_oi = sum(calls.get(s, {}).get("oi", 0) for s in otm_calls)
        otm_put_oi = sum(puts.get(s, {}).get("oi", 0) for s in otm_puts)
        
        signals["market_expectation"] = {
            "pcr_oi": round(pcr_oi, 2),
            "pcr_volume": round(pcr_vol, 2),
            "pcr_signal": pcr_signal,
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "max_pain_strike": max_pain_strike,
            "max_pain_vs_current": round((max_pain_strike - current_price) / current_price * 100, 2) if max_pain_strike else 0,
            "atm_strike": atm_strike,
            "atm_call_oi": int(atm_call_oi),
            "atm_put_oi": int(atm_put_oi),
            "otm_call_oi": int(otm_call_oi),
            "otm_put_oi": int(otm_put_oi),
            "trader_positioning": "Bullish" if otm_put_oi > otm_call_oi else ("Bearish" if otm_call_oi > otm_put_oi * 1.5 else "Neutral")
        }
        
        # ========== 2. SMART MONEY POSITIONING ==========
        
        # OI Buildup Analysis
        call_oi_buildup = sum(c["chg_oi"] for c in calls.values() if c["chg_oi"] > 0)
        call_oi_unwinding = abs(sum(c["chg_oi"] for c in calls.values() if c["chg_oi"] < 0))
        put_oi_buildup = sum(p["chg_oi"] for p in puts.values() if p["chg_oi"] > 0)
        put_oi_unwinding = abs(sum(p["chg_oi"] for p in puts.values() if p["chg_oi"] < 0))
        
        net_call_oi_change = call_oi_buildup - call_oi_unwinding
        net_put_oi_change = put_oi_buildup - put_oi_unwinding
        
        # Smart money signal interpretation
        # Price Up + Put OI Up = Bullish (puts being written as support)
        # Price Down + Call OI Up = Bearish (calls being written as resistance)
        # OI Unwinding = Trend exhaustion
        
        if put_oi_buildup > call_oi_buildup and net_put_oi_change > 0:
            smart_money_signal = "Bullish (Put Writing / Support Building)"
        elif call_oi_buildup > put_oi_buildup and net_call_oi_change > 0:
            smart_money_signal = "Bearish (Call Writing / Resistance Building)"
        elif call_oi_unwinding > call_oi_buildup or put_oi_unwinding > put_oi_buildup:
            smart_money_signal = "Trend Exhaustion (OI Unwinding)"
        else:
            smart_money_signal = "Neutral / Indecisive"
        
        # Find strikes with highest OI buildup
        call_buildup_strikes = sorted([(s, calls[s]["chg_oi"]) for s in calls if calls[s]["chg_oi"] > 0], 
                                       key=lambda x: x[1], reverse=True)[:3]
        put_buildup_strikes = sorted([(s, puts[s]["chg_oi"]) for s in puts if puts[s]["chg_oi"] > 0], 
                                      key=lambda x: x[1], reverse=True)[:3]
        
        signals["smart_money_positioning"] = {
            "call_oi_buildup": int(call_oi_buildup),
            "call_oi_unwinding": int(call_oi_unwinding),
            "net_call_oi_change": int(net_call_oi_change),
            "put_oi_buildup": int(put_oi_buildup),
            "put_oi_unwinding": int(put_oi_unwinding),
            "net_put_oi_change": int(net_put_oi_change),
            "smart_money_signal": smart_money_signal,
            "call_buildup_strikes": [{"strike": s, "oi_change": int(c)} for s, c in call_buildup_strikes],
            "put_buildup_strikes": [{"strike": s, "oi_change": int(p)} for s, p in put_buildup_strikes]
        }
        
        # ========== 3. OPTIONS-BASED SUPPORT & RESISTANCE ==========
        
        # Highest Put OI = Strong Support
        # Highest Call OI = Strong Resistance
        
        top_put_oi = sorted([(s, puts[s]["oi"]) for s in puts], key=lambda x: x[1], reverse=True)[:3]
        top_call_oi = sorted([(s, calls[s]["oi"]) for s in calls], key=lambda x: x[1], reverse=True)[:3]
        
        put_support = top_put_oi[0][0] if top_put_oi else None
        call_resistance = top_call_oi[0][0] if top_call_oi else None
        
        # OI concentration analysis
        put_concentration = top_put_oi[0][1] / total_put_oi * 100 if total_put_oi > 0 and top_put_oi else 0
        call_concentration = top_call_oi[0][1] / total_call_oi * 100 if total_call_oi > 0 and top_call_oi else 0
        
        # Expected range
        expected_range_low = put_support if put_support else current_price * 0.95
        expected_range_high = call_resistance if call_resistance else current_price * 1.05
        
        signals["options_levels"] = {
            "put_support_1": {"strike": top_put_oi[0][0], "oi": int(top_put_oi[0][1])} if len(top_put_oi) > 0 else None,
            "put_support_2": {"strike": top_put_oi[1][0], "oi": int(top_put_oi[1][1])} if len(top_put_oi) > 1 else None,
            "put_support_3": {"strike": top_put_oi[2][0], "oi": int(top_put_oi[2][1])} if len(top_put_oi) > 2 else None,
            "call_resistance_1": {"strike": top_call_oi[0][0], "oi": int(top_call_oi[0][1])} if len(top_call_oi) > 0 else None,
            "call_resistance_2": {"strike": top_call_oi[1][0], "oi": int(top_call_oi[1][1])} if len(top_call_oi) > 1 else None,
            "call_resistance_3": {"strike": top_call_oi[2][0], "oi": int(top_call_oi[2][1])} if len(top_call_oi) > 2 else None,
            "put_concentration_pct": round(put_concentration, 1),
            "call_concentration_pct": round(call_concentration, 1),
            "expected_range": {"low": expected_range_low, "high": expected_range_high},
            "range_bias": "Bullish" if current_price < (expected_range_low + expected_range_high) / 2 else "Bearish"
        }
        
        # ========== 4. VOLATILITY & RISK ==========
        
        # Get IV data for ATM options
        atm_call_iv = calls.get(atm_strike, {}).get("iv", 0)
        atm_put_iv = puts.get(atm_strike, {}).get("iv", 0)
        atm_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else max(atm_call_iv, atm_put_iv)
        
        # IV across strikes (for skew)
        otm_put_ivs = [puts[s]["iv"] for s in otm_puts if puts[s]["iv"] > 0]
        otm_call_ivs = [calls[s]["iv"] for s in otm_calls if calls[s]["iv"] > 0]
        
        avg_otm_put_iv = np.mean(otm_put_ivs) if otm_put_ivs else 0
        avg_otm_call_iv = np.mean(otm_call_ivs) if otm_call_ivs else 0
        
        # IV Skew (Put IV vs Call IV)
        iv_skew = avg_otm_put_iv - avg_otm_call_iv
        
        if iv_skew > 5:
            skew_signal = "Put Skew (Fear/Hedging)"
        elif iv_skew < -5:
            skew_signal = "Call Skew (Speculation)"
        else:
            skew_signal = "Neutral Skew"
        
        # Expected Move (ATM Straddle Price)
        atm_call_ltp = calls.get(atm_strike, {}).get("ltp", 0)
        atm_put_ltp = puts.get(atm_strike, {}).get("ltp", 0)
        straddle_price = atm_call_ltp + atm_put_ltp
        expected_move_pct = (straddle_price / current_price * 100) if current_price > 0 else 0
        
        # IV Percentile (relative to range in data)
        all_ivs = [c["iv"] for c in calls.values() if c["iv"] > 0] + [p["iv"] for p in puts.values() if p["iv"] > 0]
        iv_percentile = 50  # Default
        if all_ivs and atm_iv > 0:
            iv_percentile = sum(1 for iv in all_ivs if iv <= atm_iv) / len(all_ivs) * 100
        
        signals["volatility_risk"] = {
            "atm_iv": round(atm_iv, 2),
            "atm_call_iv": round(atm_call_iv, 2),
            "atm_put_iv": round(atm_put_iv, 2),
            "avg_otm_put_iv": round(avg_otm_put_iv, 2),
            "avg_otm_call_iv": round(avg_otm_call_iv, 2),
            "iv_skew": round(iv_skew, 2),
            "skew_signal": skew_signal,
            "straddle_price": round(straddle_price, 2),
            "expected_move_pct": round(expected_move_pct, 2),
            "expected_move_range": {
                "low": round(current_price - straddle_price, 2),
                "high": round(current_price + straddle_price, 2)
            },
            "iv_percentile": round(iv_percentile, 1),
            "iv_signal": "High IV (Expensive Options)" if iv_percentile > 70 else ("Low IV (Cheap Options)" if iv_percentile < 30 else "Normal IV")
        }
        
        # ========== 5. COMPOSITE OPTIONS SIGNAL ==========
        
        bullish_count = 0
        bearish_count = 0
        
        if pcr_oi > 1.0: bullish_count += 1
        else: bearish_count += 1
        
        if put_oi_buildup > call_oi_buildup: bullish_count += 1
        else: bearish_count += 1
        
        if current_price > max_pain_strike if max_pain_strike else False: bullish_count += 1
        elif current_price < max_pain_strike if max_pain_strike else False: bearish_count += 1
        
        if "Put Skew" in skew_signal: bearish_count += 1  # Fear in market
        elif "Call Skew" in skew_signal: bullish_count += 1  # Speculation
        
        if otm_put_oi > otm_call_oi: bullish_count += 1  # More puts being written below
        else: bearish_count += 1
        
        if bullish_count > bearish_count + 1:
            options_bias = "Bullish"
        elif bearish_count > bullish_count + 1:
            options_bias = "Bearish"
        else:
            options_bias = "Neutral"
        
        # ========== 6. CROSS-EXPIRY TREND ANALYSIS ==========
        
        # Analyze trend across expiries
        expiry_trend = {"bullish_expiries": 0, "bearish_expiries": 0}
        max_pain_trend = []
        pcr_trend = []
        
        for exp_str, data in multi_expiry_analysis.items():
            # PCR trend
            pcr_trend.append(data["pcr_oi"])
            if data["pcr_oi"] > 1.0:
                expiry_trend["bullish_expiries"] += 1
            else:
                expiry_trend["bearish_expiries"] += 1
            
            # Max pain trend
            if data["max_pain"]:
                max_pain_trend.append(data["max_pain"])
        
        # Determine cross-expiry sentiment
        if expiry_trend["bullish_expiries"] > expiry_trend["bearish_expiries"]:
            cross_expiry_sentiment = "Bullish Across Expiries"
        elif expiry_trend["bearish_expiries"] > expiry_trend["bullish_expiries"]:
            cross_expiry_sentiment = "Bearish Across Expiries"
        else:
            cross_expiry_sentiment = "Mixed Across Expiries"
        
        # Max pain direction (is it increasing or decreasing across expiries?)
        max_pain_direction = None
        if len(max_pain_trend) >= 2:
            if max_pain_trend[-1] > max_pain_trend[0]:
                max_pain_direction = "Upward (Bullish Expectation)"
            elif max_pain_trend[-1] < max_pain_trend[0]:
                max_pain_direction = "Downward (Bearish Expectation)"
            else:
                max_pain_direction = "Stable"
        
        signals["cross_expiry_analysis"] = {
            "bullish_expiries": expiry_trend["bullish_expiries"],
            "bearish_expiries": expiry_trend["bearish_expiries"],
            "cross_expiry_sentiment": cross_expiry_sentiment,
            "pcr_trend": pcr_trend,
            "avg_pcr": round(np.mean(pcr_trend), 2) if pcr_trend else 0,
            "max_pain_by_expiry": max_pain_trend,
            "max_pain_direction": max_pain_direction
        }
        
        # Add cross-expiry factors to composite
        if cross_expiry_sentiment == "Bullish Across Expiries":
            bullish_count += 1
        elif cross_expiry_sentiment == "Bearish Across Expiries":
            bearish_count += 1
        
        # Recalculate final bias with cross-expiry data
        if bullish_count > bearish_count + 2:
            options_bias = "Strong Bullish"
        elif bullish_count > bearish_count:
            options_bias = "Bullish"
        elif bearish_count > bullish_count + 2:
            options_bias = "Strong Bearish"
        elif bearish_count > bullish_count:
            options_bias = "Bearish"
        else:
            options_bias = "Neutral"
        
        signals["composite_options_signal"] = {
            "bullish_factors": bullish_count,
            "bearish_factors": bearish_count,
            "options_bias": options_bias,
            "key_support": put_support,
            "key_resistance": call_resistance,
            "trading_range": f"{expected_range_low} - {expected_range_high}",
            "max_pain_target": max_pain_strike,
            "cross_expiry_confirmation": cross_expiry_sentiment
        }
        
        return signals
    
    # ==================== FUTURES DATA SIGNALS ====================
    
    def extract_futures_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract signals from futures_data table:
        - Trend & Direction: Price trend, spot-futures premium
        - Positioning: OI buildup (long/short), unwinding
        - Leverage & Risk: OI spikes, Volume-OI divergence
        - Market Regime: Contango/Backwardation, Rollover
        """
        signals = {"available": False}
        
        # Fetch futures data
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT trade_date, expiry_date, open_price, high_price, low_price, 
                   close_price, volume, open_interest, change_in_oi
            FROM futures_data 
            WHERE symbol = ?
            ORDER BY id
        ''', (symbol.upper(),))
        
        rows = cursor.fetchall()
        if not rows or len(rows) < 5:
            return signals
        
        # Get spot price for premium/discount calculation
        cursor.execute('''
            SELECT close_price FROM price_volume_data 
            WHERE symbol = ? ORDER BY id DESC LIMIT 1
        ''', (symbol.upper(),))
        spot_row = cursor.fetchone()
        
        # Find current spot from strike range if raw price is bonus-adjusted
        all_futures_prices = []
        for r in rows:
            if not r or len(r) < 6:
                continue
            try:
                price = float(r[5])
                if price and price > 0:
                    all_futures_prices.append(price)
            except (TypeError, ValueError):
                continue
        
        if not all_futures_prices:
            return signals
        
        avg_futures = np.mean(all_futures_prices[-10:])
        spot_price = all_futures_prices[-1] * 0.995  # Default estimate
        
        if spot_row and spot_row[0]:
            raw_spot = spot_row[0]
            # Check if spot is in similar range as futures
            if avg_futures * 0.5 <= raw_spot <= avg_futures * 2:
                spot_price = raw_spot
            else:
                # Futures are on adjusted prices, estimate spot
                spot_price = all_futures_prices[-1] * 0.995  # Approximate
        
        signals["available"] = True
        signals["spot_price_ref"] = round(spot_price, 2)
        
        # Sort expiries by date
        today = datetime.now()
        expiry_data = {}
        
        for row in rows:
            if not row or len(row) < 9:
                continue
            trade_date, expiry_date, open_p, high, low, close, vol, oi, chg_oi = row
            
            if expiry_date not in expiry_data:
                expiry_data[expiry_date] = []
            
            expiry_data[expiry_date].append({
                "trade_date": trade_date,
                "open": open_p or 0,
                "high": high or 0,
                "low": low or 0,
                "close": close or 0,
                "volume": vol or 0,
                "oi": oi or 0,
                "chg_oi": chg_oi or 0
            })
        
        # Sort expiries by date proximity
        sorted_expiries = []
        for exp_str in expiry_data.keys():
            exp_dt = self._parse_expiry_date(exp_str)
            if exp_dt:
                days = (exp_dt - today).days
                sorted_expiries.append((exp_str, exp_dt, days))
        
        sorted_expiries.sort(key=lambda x: x[2])
        
        # Filter to active expiries (not expired)
        active_expiries = [(e[0], e[2]) for e in sorted_expiries if e[2] >= 0]
        
        if not active_expiries:
            # Use all if none active
            active_expiries = [(e[0], e[2]) for e in sorted_expiries[-3:]]
        
        signals["expiries_available"] = [
            {"expiry": e[0], "days_to_expiry": e[1]} for e in active_expiries[:3]
        ]
        
        # ========== NEAREST MONTH ANALYSIS ==========
        nearest_expiry = active_expiries[0][0] if active_expiries else list(expiry_data.keys())[0]
        nearest_data = expiry_data[nearest_expiry]
        
        signals["primary_expiry"] = nearest_expiry
        signals["days_to_expiry"] = active_expiries[0][1] if active_expiries else 0
        
        # Convert to arrays
        closes = np.array([d["close"] for d in nearest_data])
        volumes = np.array([d["volume"] for d in nearest_data])
        ois = np.array([d["oi"] for d in nearest_data])
        chg_ois = np.array([d["chg_oi"] for d in nearest_data])
        
        if len(closes) < 3:
            return signals
        
        current_close = closes[-1]
        current_oi = ois[-1]
        current_vol = volumes[-1]
        
        # ========== 1. TREND & DIRECTION ==========
        
        # Price trend (last 5 days)
        if len(closes) >= 5:
            price_change_5d = (closes[-1] - closes[-5]) / closes[-5] * 100
            price_trend_5d = "Up" if price_change_5d > 1 else ("Down" if price_change_5d < -1 else "Sideways")
        else:
            price_change_5d = 0
            price_trend_5d = "Unknown"
        
        # Higher highs / lower lows
        highs = np.array([d["high"] for d in nearest_data])
        lows = np.array([d["low"] for d in nearest_data])
        
        hh_count = sum(1 for i in range(1, min(5, len(highs))) if highs[-i] > highs[-i-1])
        ll_count = sum(1 for i in range(1, min(5, len(lows))) if lows[-i] < lows[-i-1])
        
        if hh_count > ll_count:
            structure = "Higher Highs (Bullish)"
        elif ll_count > hh_count:
            structure = "Lower Lows (Bearish)"
        else:
            structure = "Mixed"
        
        # Spot vs Futures Premium/Discount
        futures_premium = (current_close - spot_price) / spot_price * 100
        
        if futures_premium > 0.5:
            premium_signal = "Contango (Bullish Bias)"
        elif futures_premium < -0.5:
            premium_signal = "Backwardation (Bearish Bias)"
        else:
            premium_signal = "At Par"
        
        signals["trend_direction"] = {
            "current_futures_price": round(current_close, 2),
            "spot_price": round(spot_price, 2),
            "futures_premium_pct": round(futures_premium, 2),
            "premium_signal": premium_signal,
            "price_change_5d_pct": round(price_change_5d, 2),
            "price_trend": price_trend_5d,
            "price_structure": structure,
            "hh_count_5d": hh_count,
            "ll_count_5d": ll_count
        }
        
        # ========== 2. POSITIONING & SENTIMENT (OI Analysis) ==========
        
        # OI trend
        if len(ois) >= 5:
            oi_change_5d = (ois[-1] - ois[-5]) / ois[-5] * 100 if ois[-5] > 0 else 0
        else:
            oi_change_5d = 0
        
        # Today's OI change
        today_oi_change = chg_ois[-1] if len(chg_ois) > 0 else 0
        today_oi_change_pct = (today_oi_change / ois[-2] * 100) if len(ois) >= 2 and ois[-2] > 0 else 0
        
        # Price + OI interpretation
        price_up = closes[-1] > closes[-2] if len(closes) >= 2 else False
        oi_up = ois[-1] > ois[-2] if len(ois) >= 2 else False
        
        if price_up and oi_up:
            oi_signal = "Long Buildup (Bullish)"
            position_type = "bullish"
        elif not price_up and oi_up:
            oi_signal = "Short Buildup (Bearish)"
            position_type = "bearish"
        elif price_up and not oi_up:
            oi_signal = "Short Covering (Bullish but Weak)"
            position_type = "neutral_bullish"
        else:
            oi_signal = "Long Unwinding (Bearish)"
            position_type = "bearish"
        
        # Multi-day OI buildup analysis
        consecutive_oi_buildup = 0
        consecutive_oi_unwinding = 0
        for i in range(1, min(6, len(chg_ois))):
            if chg_ois[-i] > 0:
                consecutive_oi_buildup += 1
            else:
                break
        for i in range(1, min(6, len(chg_ois))):
            if chg_ois[-i] < 0:
                consecutive_oi_unwinding += 1
            else:
                break
        
        signals["positioning_sentiment"] = {
            "current_oi": int(current_oi),
            "today_oi_change": int(today_oi_change),
            "today_oi_change_pct": round(today_oi_change_pct, 2),
            "oi_change_5d_pct": round(oi_change_5d, 2),
            "oi_signal": oi_signal,
            "position_type": position_type,
            "consecutive_oi_buildup_days": consecutive_oi_buildup,
            "consecutive_oi_unwinding_days": consecutive_oi_unwinding,
            "oi_trend": "Building" if oi_change_5d > 5 else ("Unwinding" if oi_change_5d < -5 else "Stable")
        }
        
        # ========== 3. LEVERAGE & RISK ==========
        
        # OI spikes detection
        if len(ois) >= 20:
            oi_mean = np.mean(ois[-20:])
            oi_std = np.std(ois[-20:])
            oi_zscore = (current_oi - oi_mean) / oi_std if oi_std > 0 else 0
        else:
            oi_zscore = 0
        
        if oi_zscore > 2:
            leverage_signal = "High Leverage (Volatility Risk)"
        elif oi_zscore > 1:
            leverage_signal = "Above Average Leverage"
        else:
            leverage_signal = "Normal Leverage"
        
        # Volume-OI divergence
        if len(volumes) >= 5 and len(ois) >= 5:
            vol_trend = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) * 100 if np.mean(volumes[-5:]) > 0 else 0
            oi_trend_val = oi_change_5d
            
            if vol_trend > 20 and oi_trend_val < -5:
                vol_oi_divergence = "Volume Spike + OI Drop (Trend Exhaustion)"
            elif vol_trend < -20 and oi_trend_val > 5:
                vol_oi_divergence = "Low Volume + OI Buildup (Weak Trend)"
            elif vol_trend > 20 and oi_trend_val > 5:
                vol_oi_divergence = "Volume + OI Confirmation (Strong Trend)"
            else:
                vol_oi_divergence = "Normal"
        else:
            vol_oi_divergence = "Insufficient Data"
        
        signals["leverage_risk"] = {
            "current_volume": int(current_vol),
            "avg_volume_5d": int(np.mean(volumes[-5:])) if len(volumes) >= 5 else int(current_vol),
            "volume_ratio": round(current_vol / np.mean(volumes[-5:]), 2) if len(volumes) >= 5 and np.mean(volumes[-5:]) > 0 else 1,
            "oi_zscore": round(oi_zscore, 2),
            "leverage_signal": leverage_signal,
            "volume_oi_divergence": vol_oi_divergence
        }
        
        # ========== 4. MARKET REGIME (Multi-Expiry) ==========
        
        # Contango vs Backwardation across expiries
        expiry_prices = {}
        for exp, data in expiry_data.items():
            if data:
                latest = data[-1]
                expiry_prices[exp] = latest["close"]
        
        # Sort by expiry date
        sorted_exp_prices = []
        for exp_str, price in expiry_prices.items():
            exp_dt = self._parse_expiry_date(exp_str)
            if exp_dt:
                sorted_exp_prices.append((exp_str, exp_dt, price))
        sorted_exp_prices.sort(key=lambda x: x[1])
        
        # Check term structure
        if len(sorted_exp_prices) >= 2:
            near_price = sorted_exp_prices[0][2]
            far_price = sorted_exp_prices[-1][2]
            term_spread = (far_price - near_price) / near_price * 100
            
            if term_spread > 0.5:
                term_structure = "Contango (Normal)"
            elif term_spread < -0.5:
                term_structure = "Backwardation (Inverted)"
            else:
                term_structure = "Flat"
        else:
            term_structure = "Single Expiry"
            term_spread = 0
        
        # Rollover analysis (near expiry)
        if active_expiries and active_expiries[0][1] <= 7:
            rollover_signal = "Near Expiry - Watch for Rollover"
        elif active_expiries and active_expiries[0][1] <= 14:
            rollover_signal = "Approaching Expiry"
        else:
            rollover_signal = "Normal"
        
        signals["market_regime"] = {
            "term_structure": term_structure,
            "term_spread_pct": round(term_spread, 2),
            "expiry_prices": {e[0]: round(e[2], 2) for e in sorted_exp_prices},
            "rollover_signal": rollover_signal,
            "near_month_expiry": nearest_expiry
        }
        
        # ========== 5. MULTI-EXPIRY OI ANALYSIS ==========
        
        multi_expiry_oi = {}
        total_oi_all = 0
        
        for exp, data in expiry_data.items():
            if data:
                latest = data[-1]
                exp_dt = self._parse_expiry_date(exp)
                days = (exp_dt - today).days if exp_dt else 0
                
                multi_expiry_oi[exp] = {
                    "days_to_expiry": days,
                    "oi": int(latest["oi"]),
                    "chg_oi": int(latest["chg_oi"]),
                    "close": round(latest["close"], 2)
                }
                total_oi_all += latest["oi"]
        
        # OI concentration
        near_month_oi_pct = (current_oi / total_oi_all * 100) if total_oi_all > 0 else 100
        
        signals["multi_expiry_oi"] = {
            "by_expiry": multi_expiry_oi,
            "total_oi": int(total_oi_all),
            "near_month_oi_pct": round(near_month_oi_pct, 1),
            "oi_concentration": "High in Near Month" if near_month_oi_pct > 70 else "Spread Across Expiries"
        }
        
        # ========== 6. COMPOSITE FUTURES SIGNAL ==========
        
        bullish_count = 0
        bearish_count = 0
        
        # Trend factors
        if price_trend_5d == "Up": bullish_count += 1
        elif price_trend_5d == "Down": bearish_count += 1
        
        if "Higher Highs" in structure: bullish_count += 1
        elif "Lower Lows" in structure: bearish_count += 1
        
        # Premium factor
        if "Contango" in premium_signal: bullish_count += 1
        elif "Backwardation" in premium_signal: bearish_count += 1
        
        # OI factors
        if position_type == "bullish": bullish_count += 2
        elif position_type == "bearish": bearish_count += 2
        elif position_type == "neutral_bullish": bullish_count += 1
        
        # Term structure
        if "Contango" in term_structure: bullish_count += 1
        elif "Backwardation" in term_structure: bearish_count += 1
        
        if bullish_count > bearish_count + 2:
            futures_bias = "Strong Bullish"
        elif bullish_count > bearish_count:
            futures_bias = "Bullish"
        elif bearish_count > bullish_count + 2:
            futures_bias = "Strong Bearish"
        elif bearish_count > bullish_count:
            futures_bias = "Bearish"
        else:
            futures_bias = "Neutral"
        
        signals["composite_futures_signal"] = {
            "bullish_factors": bullish_count,
            "bearish_factors": bearish_count,
            "futures_bias": futures_bias,
            "key_level": round(current_close, 2),
            "oi_signal": oi_signal,
            "trend_confirmation": "Yes" if futures_bias == "Bullish" and price_trend_5d == "Up" else ("Yes" if futures_bias == "Bearish" and price_trend_5d == "Down" else "No")
        }
        
        return signals
    
    # ==================== SECTOR / LAYER 2 SIGNALS ====================
    
    def extract_sector_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract Layer 2 sector signals and correlate with equity:
        - Sector trend & momentum
        - Sector volatility & strength
        - Relative strength (Stock vs Sector)
        - Divergence / confirmation signals
        """
        signals = {"available": False}
        
        # Fetch sector index data
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT index_name, trade_date, open_value, high_value, low_value, 
                   close_value, volume, turnover
            FROM sector_index_history
            ORDER BY id
        ''')
        
        sector_rows = cursor.fetchall()
        if len(sector_rows) < 10:
            return signals
        
        # Get equity price-volume data for correlation
        cursor.execute('''
            SELECT trade_date, open_price, high_price, low_price, close_price, 
                   volume, delivery_pct
            FROM price_volume_data 
            WHERE symbol = ?
            ORDER BY id DESC
            LIMIT 100
        ''', (symbol.upper(),))
        
        equity_rows = list(reversed(cursor.fetchall()))
        
        if len(equity_rows) < 20:
            return signals
        
        # Extract sector name
        sector_name = sector_rows[0][0] if sector_rows else "Unknown"
        
        signals["available"] = True
        signals["sector_name"] = sector_name
        
        # Convert sector data to arrays
        sector_dates = [r[1] for r in sector_rows]
        sector_opens = np.array([r[2] or 0 for r in sector_rows])
        sector_highs = np.array([r[3] or 0 for r in sector_rows])
        sector_lows = np.array([r[4] or 0 for r in sector_rows])
        sector_closes = np.array([r[5] or 0 for r in sector_rows])
        sector_volumes = np.array([r[6] or 0 for r in sector_rows])
        sector_turnovers = np.array([r[7] or 0 for r in sector_rows])
        
        # Convert equity data to arrays
        equity_dates = [r[0] for r in equity_rows]
        equity_closes = np.array([r[4] or 0 for r in equity_rows])
        equity_volumes = np.array([r[5] or 0 for r in equity_rows])
        equity_delivery = np.array([r[6] or 0 for r in equity_rows])
        
        signals["data_points"] = {
            "sector": len(sector_rows),
            "equity": len(equity_rows)
        }
        
        # ========== 1. SECTOR TREND & MOMENTUM ==========
        
        current_sector = sector_closes[-1]
        
        # SMAs for sector
        sector_sma_20 = np.mean(sector_closes[-20:]) if len(sector_closes) >= 20 else np.mean(sector_closes)
        sector_sma_50 = np.mean(sector_closes[-50:]) if len(sector_closes) >= 50 else np.mean(sector_closes)
        
        # Sector trend
        if current_sector > sector_sma_20 > sector_sma_50:
            sector_trend = "Strong Uptrend"
        elif current_sector > sector_sma_20:
            sector_trend = "Uptrend"
        elif current_sector < sector_sma_20 < sector_sma_50:
            sector_trend = "Strong Downtrend"
        elif current_sector < sector_sma_20:
            sector_trend = "Downtrend"
        else:
            sector_trend = "Sideways"
        
        # Sector momentum (returns)
        if len(sector_closes) >= 5:
            sector_return_5d = (sector_closes[-1] - sector_closes[-5]) / sector_closes[-5] * 100
        else:
            sector_return_5d = 0
        
        if len(sector_closes) >= 20:
            sector_return_20d = (sector_closes[-1] - sector_closes[-20]) / sector_closes[-20] * 100
        else:
            sector_return_20d = 0
        
        # Sector RSI
        sector_returns = np.diff(sector_closes) / sector_closes[:-1]
        sector_gains = np.where(sector_returns > 0, sector_returns, 0)
        sector_losses = np.where(sector_returns < 0, -sector_returns, 0)
        
        if len(sector_gains) >= 14:
            avg_gain = np.mean(sector_gains[-14:])
            avg_loss = np.mean(sector_losses[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            sector_rsi = 100 - (100 / (1 + rs))
        else:
            sector_rsi = 50
        
        signals["sector_trend_momentum"] = {
            "current_value": round(current_sector, 2),
            "sma_20": round(sector_sma_20, 2),
            "sma_50": round(sector_sma_50, 2),
            "trend": sector_trend,
            "return_5d_pct": round(sector_return_5d, 2),
            "return_20d_pct": round(sector_return_20d, 2),
            "rsi_14": round(sector_rsi, 2),
            "momentum": "Strong" if sector_return_5d > 3 else ("Weak" if sector_return_5d < -3 else "Normal")
        }
        
        # ========== 2. SECTOR VOLATILITY & STRENGTH ==========
        
        # ATR for sector
        sector_tr = np.maximum(sector_highs[1:] - sector_lows[1:],
                              np.maximum(np.abs(sector_highs[1:] - sector_closes[:-1]),
                                        np.abs(sector_lows[1:] - sector_closes[:-1])))
        sector_atr = np.mean(sector_tr[-14:]) if len(sector_tr) >= 14 else np.mean(sector_tr) if len(sector_tr) > 0 else 0
        
        # Volatility as % of price
        sector_volatility_pct = (sector_atr / current_sector * 100) if current_sector > 0 else 0
        
        # Sector strength (bullish vs bearish days)
        bullish_days = sum(1 for i in range(-20, 0) if i < len(sector_closes) and sector_closes[i] > sector_opens[i])
        
        # Volume trend
        if len(sector_volumes) >= 20:
            avg_sector_vol = np.mean(sector_volumes[-20:])
            current_sector_vol = sector_volumes[-1]
            sector_vol_ratio = current_sector_vol / avg_sector_vol if avg_sector_vol > 0 else 1
        else:
            sector_vol_ratio = 1
        
        signals["sector_volatility_strength"] = {
            "atr_14": round(sector_atr, 2),
            "volatility_pct": round(sector_volatility_pct, 2),
            "bullish_days_20": bullish_days,
            "bearish_days_20": 20 - bullish_days,
            "market_control": "Bulls" if bullish_days > 12 else ("Bears" if bullish_days < 8 else "Contested"),
            "volume_ratio": round(sector_vol_ratio, 2),
            "volume_signal": "Expansion" if sector_vol_ratio > 1.3 else ("Contraction" if sector_vol_ratio < 0.7 else "Normal")
        }
        
        # ========== 3. SECTOR BREAKOUT / BREAKDOWN ==========
        
        if len(sector_closes) >= 20:
            sector_20d_high = max(sector_highs[-20:])
            sector_20d_low = min(sector_lows[-20:])
            
            if current_sector >= sector_20d_high * 0.99:
                sector_breakout = "20-Day Breakout"
            elif current_sector <= sector_20d_low * 1.01:
                sector_breakout = "20-Day Breakdown"
            else:
                sector_breakout = "Within Range"
        else:
            sector_breakout = "Insufficient Data"
            sector_20d_high = current_sector
            sector_20d_low = current_sector
        
        signals["sector_levels"] = {
            "high_20d": round(sector_20d_high, 2),
            "low_20d": round(sector_20d_low, 2),
            "breakout_status": sector_breakout,
            "distance_from_high_pct": round((sector_20d_high - current_sector) / current_sector * 100, 2),
            "distance_from_low_pct": round((current_sector - sector_20d_low) / sector_20d_low * 100, 2) if sector_20d_low > 0 else 0
        }
        
        # ========== 4. RELATIVE STRENGTH (Stock vs Sector) ==========
        
        # Align dates for correlation (use last N common days)
        min_len = min(len(sector_closes), len(equity_closes), 50)
        
        sector_for_corr = sector_closes[-min_len:]
        equity_for_corr = equity_closes[-min_len:]
        
        # Calculate returns for correlation
        sector_rets = np.diff(sector_for_corr) / sector_for_corr[:-1]
        equity_rets = np.diff(equity_for_corr) / equity_for_corr[:-1]
        
        # Correlation coefficient
        if len(sector_rets) >= 5 and len(equity_rets) >= 5:
            correlation = np.corrcoef(sector_rets[:len(equity_rets)], equity_rets[:len(sector_rets)])[0, 1]
        else:
            correlation = 0
        
        # Relative strength: Stock return vs Sector return
        equity_return_20d = (equity_closes[-1] - equity_closes[-20]) / equity_closes[-20] * 100 if len(equity_closes) >= 20 else 0
        relative_strength = equity_return_20d - sector_return_20d
        
        if relative_strength > 5:
            rs_signal = "Strong Outperformer"
        elif relative_strength > 0:
            rs_signal = "Outperformer"
        elif relative_strength < -5:
            rs_signal = "Strong Underperformer"
        elif relative_strength < 0:
            rs_signal = "Underperformer"
        else:
            rs_signal = "Inline with Sector"
        
        signals["relative_strength"] = {
            "stock_return_20d_pct": round(equity_return_20d, 2),
            "sector_return_20d_pct": round(sector_return_20d, 2),
            "relative_strength_pct": round(relative_strength, 2),
            "rs_signal": rs_signal,
            "correlation": round(correlation, 2) if not np.isnan(correlation) else 0,
            "correlation_strength": "High" if abs(correlation) > 0.7 else ("Moderate" if abs(correlation) > 0.4 else "Low")
        }
        
        # ========== 5. CONFIRMATION / DIVERGENCE SIGNALS ==========
        
        # Stock trend
        equity_sma_20 = np.mean(equity_closes[-20:]) if len(equity_closes) >= 20 else np.mean(equity_closes)
        stock_trending_up = equity_closes[-1] > equity_sma_20
        sector_trending_up = current_sector > sector_sma_20
        
        # Confirmation signal
        if stock_trending_up and sector_trending_up:
            trend_alignment = "Bullish Confirmation (Both Up)"
        elif not stock_trending_up and not sector_trending_up:
            trend_alignment = "Bearish Confirmation (Both Down)"
        elif stock_trending_up and not sector_trending_up:
            trend_alignment = "Bullish Divergence (Stock Up, Sector Down)"
        else:
            trend_alignment = "Bearish Divergence (Stock Down, Sector Up)"
        
        # Volume alignment
        equity_vol_ratio = equity_volumes[-1] / np.mean(equity_volumes[-20:]) if len(equity_volumes) >= 20 and np.mean(equity_volumes[-20:]) > 0 else 1
        
        if sector_vol_ratio > 1.2 and equity_vol_ratio > 1.2:
            volume_alignment = "Confirmed (Both Expanding)"
        elif sector_vol_ratio < 0.8 and equity_vol_ratio < 0.8:
            volume_alignment = "Confirmed (Both Contracting)"
        else:
            volume_alignment = "Divergent"
        
        # Breakout confirmation
        equity_20d_high = max(equity_closes[-20:]) if len(equity_closes) >= 20 else equity_closes[-1]
        stock_at_breakout = equity_closes[-1] >= equity_20d_high * 0.99
        
        if stock_at_breakout and sector_breakout == "20-Day Breakout":
            breakout_confirmation = "High Conviction (Both Breaking Out)"
        elif stock_at_breakout and sector_breakout != "20-Day Breakout":
            breakout_confirmation = "Stock Leading (Sector Lagging)"
        elif not stock_at_breakout and sector_breakout == "20-Day Breakout":
            breakout_confirmation = "Sector Leading (Stock Lagging)"
        else:
            breakout_confirmation = "No Breakout"
        
        signals["confirmation_divergence"] = {
            "trend_alignment": trend_alignment,
            "volume_alignment": volume_alignment,
            "breakout_confirmation": breakout_confirmation,
            "trade_conviction": "High" if "Confirmation" in trend_alignment and "Confirmed" in volume_alignment else "Low"
        }
        
        # ========== 6. SECTOR ROTATION / MARKET CONTEXT ==========
        
        # Sector leadership (based on momentum)
        if sector_return_20d > 5 and sector_rsi > 60:
            sector_status = "Leader (Strong Momentum)"
        elif sector_return_20d > 0 and sector_rsi > 50:
            sector_status = "Participant (Positive Momentum)"
        elif sector_return_20d < -5 and sector_rsi < 40:
            sector_status = "Laggard (Weak Momentum)"
        else:
            sector_status = "Neutral"
        
        # Risk-on / Risk-off (based on volatility and trend)
        if sector_trend == "Strong Uptrend" and sector_volatility_pct < 2:
            risk_regime = "Risk-On (Trending Up, Low Vol)"
        elif sector_trend == "Strong Downtrend" or sector_volatility_pct > 3:
            risk_regime = "Risk-Off (High Vol or Downtrend)"
        else:
            risk_regime = "Neutral"
        
        signals["market_context"] = {
            "sector_status": sector_status,
            "risk_regime": risk_regime,
            "sector_leadership": "Yes" if "Leader" in sector_status else "No",
            "rotation_signal": "Money Flowing In" if sector_vol_ratio > 1.3 and sector_return_5d > 0 else ("Money Flowing Out" if sector_vol_ratio > 1.3 and sector_return_5d < 0 else "Stable")
        }
        
        # ========== 7. COMPOSITE SECTOR SIGNAL ==========
        
        bullish_count = 0
        bearish_count = 0
        
        if "Uptrend" in sector_trend: bullish_count += 1
        elif "Downtrend" in sector_trend: bearish_count += 1
        
        if sector_rsi > 50: bullish_count += 1
        elif sector_rsi < 50: bearish_count += 1
        
        if "Outperformer" in rs_signal: bullish_count += 1
        elif "Underperformer" in rs_signal: bearish_count += 1
        
        if "Bullish" in trend_alignment: bullish_count += 1
        elif "Bearish" in trend_alignment: bearish_count += 1
        
        if sector_breakout == "20-Day Breakout": bullish_count += 1
        elif sector_breakout == "20-Day Breakdown": bearish_count += 1
        
        if "Leader" in sector_status: bullish_count += 1
        elif "Laggard" in sector_status: bearish_count += 1
        
        if bullish_count > bearish_count + 2:
            sector_bias = "Strong Bullish"
        elif bullish_count > bearish_count:
            sector_bias = "Bullish"
        elif bearish_count > bullish_count + 2:
            sector_bias = "Strong Bearish"
        elif bearish_count > bullish_count:
            sector_bias = "Bearish"
        else:
            sector_bias = "Neutral"
        
        signals["composite_sector_signal"] = {
            "bullish_factors": bullish_count,
            "bearish_factors": bearish_count,
            "sector_bias": sector_bias,
            "relative_strength": rs_signal,
            "trade_setup": "High Probability" if breakout_confirmation == "High Conviction (Both Breaking Out)" else "Standard"
        }
        
        return signals
    
    # ==================== INTRADAY SECTOR SIGNALS ====================
    
    def extract_intraday_sector_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract intraday sector signals from layer2_sectors and correlate with equity intraday data:
        - Intraday trend & bias
        - VWAP-like behavior
        - Momentum shifts
        - Intraday volatility regime
        - Time-based patterns
        - Stock vs Sector relative strength
        - Entry timing & risk management signals
        """
        signals = {"available": False}
        
        # Fetch sector intraday data from layer2_sectors
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT sector_name, json_data 
            FROM layer2_sectors 
            WHERE symbol = ?
        ''', (symbol.upper(),))
        
        sector_row = cursor.fetchone()
        if not sector_row:
            return signals
        
        import json
        sector_name = sector_row[0]
        sector_data = json.loads(sector_row[1])

        # Safely extract nested data - handle None values
        data_obj = sector_data.get('data') if sector_data else None
        if data_obj is None:
            return signals

        # Extract grapthData (intraday price points)
        sector_graph = data_obj.get('grapthData', []) if data_obj else []
        sector_close_price = data_obj.get('closePrice', 0) if data_obj else 0
        
        if len(sector_graph) < 30:
            return signals
        
        # Fetch equity intraday data
        intraday_json = self._get_layer1_json(symbol, "intraday_data")
        if not intraday_json:
            return signals
        
        equity_graph = intraday_json.get('grapthData', [])
        equity_close_price = intraday_json.get('closePrice', 0)
        
        if len(equity_graph) < 30:
            return signals
        
        signals["available"] = True
        signals["sector_name"] = sector_name
        
        # Parse sector data: [timestamp, price, status]
        # Filter only normal market data (NM) for analysis, but keep PO for open
        sector_prices = []
        sector_times = []
        for point in sector_graph:
            ts, price, status = point[0], point[1], point[2] if len(point) > 2 else 'NM'
            sector_prices.append(price)
            sector_times.append(ts)
        
        sector_prices = np.array(sector_prices)
        sector_times = np.array(sector_times)
        
        # Parse equity data
        equity_prices = []
        equity_times = []
        for point in equity_graph:
            ts, price, status = point[0], point[1], point[2] if len(point) > 2 else 'NM'
            equity_prices.append(price)
            equity_times.append(ts)
        
        equity_prices = np.array(equity_prices)
        equity_times = np.array(equity_times)
        
        signals["data_points"] = {
            "sector": len(sector_prices),
            "equity": len(equity_prices)
        }
        
        # ========== 1. INTRADAY TREND & BIAS ==========
        
        sector_open = sector_prices[0]
        sector_high = np.max(sector_prices)
        sector_low = np.min(sector_prices)
        sector_current = sector_prices[-1]
        sector_range = sector_high - sector_low
        
        # Position within range
        if sector_range > 0:
            sector_range_position = (sector_current - sector_low) / sector_range * 100
        else:
            sector_range_position = 50
        
        # Trend determination
        sector_vwap_approx = np.mean(sector_prices)  # Approximate VWAP (no volume)
        
        if sector_current > sector_vwap_approx and sector_current > sector_open:
            sector_intraday_trend = "Bullish"
        elif sector_current < sector_vwap_approx and sector_current < sector_open:
            sector_intraday_trend = "Bearish"
        else:
            sector_intraday_trend = "Range-bound"
        
        # Trend strength based on distance from VWAP
        vwap_distance_pct = (sector_current - sector_vwap_approx) / sector_vwap_approx * 100 if sector_vwap_approx > 0 else 0
        
        signals["sector_intraday_trend"] = {
            "open": round(sector_open, 2),
            "high": round(sector_high, 2),
            "low": round(sector_low, 2),
            "current": round(sector_current, 2),
            "change_pct": round((sector_current - sector_open) / sector_open * 100, 2) if sector_open > 0 else 0,
            "range_position_pct": round(sector_range_position, 2),
            "trend": sector_intraday_trend,
            "bias": "Strong Bullish" if sector_range_position > 80 else ("Strong Bearish" if sector_range_position < 20 else "Neutral")
        }
        
        # ========== 2. VWAP-LIKE BEHAVIOR ==========
        
        # Calculate running VWAP approximation (mean price up to each point)
        cumulative_sum = np.cumsum(sector_prices)
        cumulative_count = np.arange(1, len(sector_prices) + 1)
        running_vwap = cumulative_sum / cumulative_count
        
        # Count bars above/below VWAP
        above_vwap = np.sum(sector_prices > running_vwap)
        below_vwap = np.sum(sector_prices < running_vwap)
        
        # Current position relative to VWAP
        current_vwap = running_vwap[-1]
        vwap_signal = "Above VWAP" if sector_current > current_vwap else "Below VWAP"
        
        # VWAP reclaim/rejection
        if len(sector_prices) >= 20:
            recent_prices = sector_prices[-20:]
            recent_vwap = running_vwap[-20:]
            crosses_up = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_vwap[i] and recent_prices[i-1] <= recent_vwap[i-1])
            crosses_down = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_vwap[i] and recent_prices[i-1] >= recent_vwap[i-1])
        else:
            crosses_up = crosses_down = 0
        
        signals["vwap_behavior"] = {
            "current_vwap": round(current_vwap, 2),
            "price_vs_vwap": vwap_signal,
            "vwap_distance_pct": round(vwap_distance_pct, 2),
            "bars_above_vwap": int(above_vwap),
            "bars_below_vwap": int(below_vwap),
            "vwap_dominance": "Bulls" if above_vwap > below_vwap * 1.5 else ("Bears" if below_vwap > above_vwap * 1.5 else "Balanced"),
            "recent_vwap_crosses_up": crosses_up,
            "recent_vwap_crosses_down": crosses_down,
            "vwap_trend": "Holding Above" if crosses_up > crosses_down and vwap_signal == "Above VWAP" else ("Holding Below" if crosses_down > crosses_up and vwap_signal == "Below VWAP" else "Volatile Around VWAP")
        }
        
        # ========== 3. MOMENTUM SHIFTS ==========
        
        # Calculate momentum using price changes
        if len(sector_prices) >= 10:
            price_changes = np.diff(sector_prices)
            
            # Split into segments for momentum analysis
            segment_size = len(price_changes) // 4
            if segment_size > 0:
                seg1 = np.sum(price_changes[:segment_size])
                seg2 = np.sum(price_changes[segment_size:2*segment_size])
                seg3 = np.sum(price_changes[2*segment_size:3*segment_size])
                seg4 = np.sum(price_changes[3*segment_size:])
                
                # Determine momentum phase
                if seg4 > 0 and seg4 > seg3:
                    momentum_phase = "Impulse Up (Accelerating)"
                elif seg4 > 0 and seg4 < seg3:
                    momentum_phase = "Impulse Up (Decelerating)"
                elif seg4 < 0 and seg4 < seg3:
                    momentum_phase = "Impulse Down (Accelerating)"
                elif seg4 < 0 and seg4 > seg3:
                    momentum_phase = "Mean Reversion (Bounce)"
                else:
                    momentum_phase = "Consolidation"
                
                # Recent momentum (last 20%)
                recent_momentum = np.sum(price_changes[-len(price_changes)//5:]) if len(price_changes) >= 5 else 0
            else:
                momentum_phase = "Insufficient Data"
                recent_momentum = 0
                seg1 = seg2 = seg3 = seg4 = 0
        else:
            momentum_phase = "Insufficient Data"
            recent_momentum = 0
            seg1 = seg2 = seg3 = seg4 = 0
        
        signals["momentum_analysis"] = {
            "momentum_phase": momentum_phase,
            "segment_1_change": round(seg1, 2),
            "segment_2_change": round(seg2, 2),
            "segment_3_change": round(seg3, 2),
            "segment_4_change": round(seg4, 2),
            "recent_momentum": round(recent_momentum, 2),
            "momentum_direction": "Bullish" if recent_momentum > 0 else ("Bearish" if recent_momentum < 0 else "Flat")
        }
        
        # ========== 4. INTRADAY VOLATILITY REGIME ==========
        
        # Calculate rolling volatility
        if len(sector_prices) >= 20:
            returns = np.diff(sector_prices) / sector_prices[:-1]
            
            # Split session into halves
            mid = len(returns) // 2
            first_half_vol = np.std(returns[:mid]) * 100 if mid > 0 else 0
            second_half_vol = np.std(returns[mid:]) * 100 if mid > 0 else 0
            overall_vol = np.std(returns) * 100
            
            # Volatility regime
            if second_half_vol > first_half_vol * 1.5:
                vol_regime = "Expansion (Volatility Increasing)"
            elif second_half_vol < first_half_vol * 0.7:
                vol_regime = "Contraction (Volatility Decreasing)"
            else:
                vol_regime = "Stable"
            
            # ATR-like metric
            intraday_atr = sector_range / sector_open * 100 if sector_open > 0 else 0
        else:
            first_half_vol = second_half_vol = overall_vol = 0
            vol_regime = "Insufficient Data"
            intraday_atr = 0
        
        signals["volatility_regime"] = {
            "intraday_range_pct": round(intraday_atr, 2),
            "first_half_volatility": round(first_half_vol, 4),
            "second_half_volatility": round(second_half_vol, 4),
            "overall_volatility": round(overall_vol, 4),
            "regime": vol_regime,
            "trading_environment": "Trending" if intraday_atr > 1.5 else ("Choppy" if intraday_atr < 0.5 else "Normal")
        }
        
        # ========== 5. HIGH-LOW STRUCTURE (Trend Day vs Choppy) ==========
        
        # Count higher highs and lower lows in segments
        if len(sector_prices) >= 20:
            segment_len = len(sector_prices) // 5
            segment_highs = []
            segment_lows = []
            
            for i in range(5):
                start = i * segment_len
                end = (i + 1) * segment_len if i < 4 else len(sector_prices)
                segment_highs.append(np.max(sector_prices[start:end]))
                segment_lows.append(np.min(sector_prices[start:end]))
            
            # Count higher highs and lower lows
            higher_highs = sum(1 for i in range(1, len(segment_highs)) if segment_highs[i] > segment_highs[i-1])
            lower_lows = sum(1 for i in range(1, len(segment_lows)) if segment_lows[i] < segment_lows[i-1])
            higher_lows = sum(1 for i in range(1, len(segment_lows)) if segment_lows[i] > segment_lows[i-1])
            lower_highs = sum(1 for i in range(1, len(segment_highs)) if segment_highs[i] < segment_highs[i-1])
            
            # Day type
            if higher_highs >= 3 and higher_lows >= 2:
                day_type = "Trend Day Up"
            elif lower_lows >= 3 and lower_highs >= 2:
                day_type = "Trend Day Down"
            elif higher_highs >= 2 and lower_lows >= 2:
                day_type = "Choppy/Range Day"
            else:
                day_type = "Mixed"
        else:
            higher_highs = lower_lows = higher_lows = lower_highs = 0
            day_type = "Insufficient Data"
        
        signals["day_structure"] = {
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "day_type": day_type,
            "structure_quality": "Clean Trend" if day_type.startswith("Trend Day") else ("Noisy" if day_type == "Choppy/Range Day" else "Mixed")
        }
        
        # ========== 6. TIME-BASED PATTERNS ==========
        
        # Split into opening (first 20%), mid-day (middle 60%), closing (last 20%)
        n = len(sector_prices)
        opening_end = n // 5
        closing_start = 4 * n // 5
        
        opening_prices = sector_prices[:opening_end] if opening_end > 0 else sector_prices[:5]
        midday_prices = sector_prices[opening_end:closing_start]
        closing_prices = sector_prices[closing_start:]
        
        opening_change = (opening_prices[-1] - opening_prices[0]) / opening_prices[0] * 100 if len(opening_prices) > 1 and opening_prices[0] > 0 else 0
        midday_change = (midday_prices[-1] - midday_prices[0]) / midday_prices[0] * 100 if len(midday_prices) > 1 and midday_prices[0] > 0 else 0
        closing_change = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] * 100 if len(closing_prices) > 1 and closing_prices[0] > 0 else 0
        
        # Time patterns
        if opening_change > 0.2:
            opening_pattern = "Opening Drive Up"
        elif opening_change < -0.2:
            opening_pattern = "Opening Drive Down"
        else:
            opening_pattern = "Flat Open"
        
        if midday_change < -0.1 and opening_change > 0:
            midday_pattern = "Mid-Day Fade"
        elif midday_change > 0.1 and opening_change < 0:
            midday_pattern = "Mid-Day Recovery"
        else:
            midday_pattern = "Continuation"
        
        if closing_change > 0.1:
            closing_pattern = "Closing Push Up"
        elif closing_change < -0.1:
            closing_pattern = "Closing Weakness"
        else:
            closing_pattern = "Flat Close"
        
        signals["time_patterns"] = {
            "opening_change_pct": round(opening_change, 2),
            "midday_change_pct": round(midday_change, 2),
            "closing_change_pct": round(closing_change, 2),
            "opening_pattern": opening_pattern,
            "midday_pattern": midday_pattern,
            "closing_pattern": closing_pattern,
            "session_flow": f"{opening_pattern} → {midday_pattern} → {closing_pattern}"
        }
        
        # ========== 7. STOCK VS SECTOR RELATIVE STRENGTH (Intraday) ==========
        
        # Align data by finding common timestamps
        equity_open = equity_prices[0]
        equity_current = equity_prices[-1]
        equity_high = np.max(equity_prices)
        equity_low = np.min(equity_prices)
        
        # Calculate returns
        stock_return = (equity_current - equity_open) / equity_open * 100 if equity_open > 0 else 0
        sector_return = (sector_current - sector_open) / sector_open * 100 if sector_open > 0 else 0
        
        # Relative strength
        intraday_rs = stock_return - sector_return
        
        if stock_return > 0 and sector_return < 0:
            rs_signal = "Stock-Specific Strength (Stock Up, Sector Down)"
        elif stock_return < 0 and sector_return > 0:
            rs_signal = "Stock-Specific Weakness (Stock Down, Sector Up)"
        elif stock_return > sector_return + 0.3:
            rs_signal = "Outperforming Sector"
        elif stock_return < sector_return - 0.3:
            rs_signal = "Underperforming Sector"
        else:
            rs_signal = "Inline with Sector"
        
        signals["intraday_relative_strength"] = {
            "stock_return_pct": round(stock_return, 2),
            "sector_return_pct": round(sector_return, 2),
            "relative_strength_pct": round(intraday_rs, 2),
            "rs_signal": rs_signal,
            "alpha_generation": "Positive Alpha" if intraday_rs > 0.5 else ("Negative Alpha" if intraday_rs < -0.5 else "Neutral")
        }
        
        # ========== 8. INTRADAY BETA ==========
        
        # Calculate intraday beta using returns correlation
        min_len = min(len(sector_prices), len(equity_prices))
        if min_len >= 20:
            sector_rets = np.diff(sector_prices[-min_len:]) / sector_prices[-min_len:-1]
            equity_rets = np.diff(equity_prices[-min_len:]) / equity_prices[-min_len:-1]
            
            # Remove any inf/nan
            valid_mask = np.isfinite(sector_rets) & np.isfinite(equity_rets)
            sector_rets = sector_rets[valid_mask]
            equity_rets = equity_rets[valid_mask]
            
            if len(sector_rets) > 10:
                # Beta = Cov(stock, sector) / Var(sector)
                covariance = np.cov(equity_rets, sector_rets)[0, 1]
                sector_variance = np.var(sector_rets)
                intraday_beta = covariance / sector_variance if sector_variance > 0 else 1
                
                # Correlation
                correlation = np.corrcoef(equity_rets, sector_rets)[0, 1]
            else:
                intraday_beta = 1
                correlation = 0
        else:
            intraday_beta = 1
            correlation = 0
        
        signals["intraday_beta"] = {
            "beta": round(intraday_beta, 2) if not np.isnan(intraday_beta) else 1.0,
            "correlation": round(correlation, 2) if not np.isnan(correlation) else 0,
            "sensitivity": "High Beta" if intraday_beta > 1.3 else ("Low Beta" if intraday_beta < 0.7 else "Normal Beta"),
            "interpretation": f"Stock moves {round(intraday_beta, 2)}x sector moves"
        }
        
        # ========== 9. ENTRY TIMING & RISK MANAGEMENT ==========
        
        # Entry timing based on sector momentum windows
        if sector_intraday_trend == "Bullish" and momentum_phase.startswith("Impulse Up"):
            entry_timing = "Favorable Long Entry (Sector momentum up)"
        elif sector_intraday_trend == "Bearish" and momentum_phase.startswith("Impulse Down"):
            entry_timing = "Favorable Short Entry (Sector momentum down)"
        elif "Mean Reversion" in momentum_phase:
            entry_timing = "Reversal Play Possible"
        else:
            entry_timing = "Wait for Clarity"
        
        # Breakout confirmation
        equity_vwap_approx = np.mean(equity_prices)
        stock_above_vwap = equity_current > equity_vwap_approx
        sector_above_vwap = sector_current > current_vwap
        
        if stock_above_vwap and sector_above_vwap:
            breakout_quality = "High Conviction (Both above VWAP)"
        elif stock_above_vwap and not sector_above_vwap:
            breakout_quality = "Low Quality (Stock leading, Sector weak)"
        elif not stock_above_vwap and sector_above_vwap:
            breakout_quality = "Stock Lagging Sector"
        else:
            breakout_quality = "Both Weak"
        
        # Risk management
        if sector_intraday_trend == "Bearish" and sector_range_position < 30:
            risk_signal = "Avoid Longs (Sector bearish)"
        elif sector_intraday_trend == "Bullish" and sector_range_position > 70:
            risk_signal = "Avoid Shorts (Sector bullish)"
        elif vol_regime == "Expansion (Volatility Increasing)":
            risk_signal = "Reduce Position Size (Volatility expanding)"
        else:
            risk_signal = "Normal Risk Parameters"
        
        signals["trading_signals"] = {
            "entry_timing": entry_timing,
            "breakout_quality": breakout_quality,
            "risk_signal": risk_signal,
            "sector_support": sector_intraday_trend == "Bullish",
            "momentum_aligned": "Impulse" in momentum_phase
        }
        
        # ========== 10. COMPOSITE INTRADAY SECTOR SIGNAL ==========
        
        bullish_count = 0
        bearish_count = 0
        
        if sector_intraday_trend == "Bullish": bullish_count += 1
        elif sector_intraday_trend == "Bearish": bearish_count += 1
        
        if vwap_signal == "Above VWAP": bullish_count += 1
        else: bearish_count += 1
        
        if "Up" in momentum_phase: bullish_count += 1
        elif "Down" in momentum_phase: bearish_count += 1
        
        if day_type == "Trend Day Up": bullish_count += 1
        elif day_type == "Trend Day Down": bearish_count += 1
        
        if closing_change > 0: bullish_count += 1
        elif closing_change < 0: bearish_count += 1
        
        if "Outperforming" in rs_signal or "Stock-Specific Strength" in rs_signal: bullish_count += 1
        elif "Underperforming" in rs_signal or "Stock-Specific Weakness" in rs_signal: bearish_count += 1
        
        if bullish_count > bearish_count + 2:
            composite_bias = "Strong Bullish"
        elif bullish_count > bearish_count:
            composite_bias = "Bullish"
        elif bearish_count > bullish_count + 2:
            composite_bias = "Strong Bearish"
        elif bearish_count > bullish_count:
            composite_bias = "Bearish"
        else:
            composite_bias = "Neutral"
        
        signals["composite_intraday_signal"] = {
            "bullish_factors": bullish_count,
            "bearish_factors": bearish_count,
            "intraday_bias": composite_bias,
            "sector_support": sector_intraday_trend,
            "trade_recommendation": entry_timing
        }
        
        return signals
    
    # ==================== MARKET STATUS SIGNALS (Layer 3) ====================
    
    def extract_market_status_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract market status signals from layer3_data (market_status):
        - GIFT NIFTY overnight sentiment
        - Market regime & tradability
        - Index-level risk tone
        - Asset-class flow hints
        - Macro backdrop (market cap)
        """
        signals = {"available": False}
        
        # Fetch market_status data
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT json_data FROM layer3_data 
            WHERE data_type = 'market_status'
        ''')
        
        row = cursor.fetchone()
        if not row:
            return signals
        
        import json
        data = json.loads(row[0])
        
        market_states = data.get("marketState", [])
        market_cap = data.get("marketcap", {})
        indicative_nifty = data.get("indicativenifty50", {})
        gift_nifty = data.get("giftnifty", {})
        
        if not gift_nifty and not indicative_nifty:
            return signals
        
        signals["available"] = True
        signals["data_timestamp"] = gift_nifty.get("TIMESTMP", indicative_nifty.get("dateTime", "N/A"))
        
        # ========== 1. GIFT NIFTY - OVERNIGHT SENTIMENT ==========
        
        gift_last = float(gift_nifty.get("LASTPRICE", 0)) if gift_nifty.get("LASTPRICE") else 0
        gift_change = float(gift_nifty.get("DAYCHANGE", 0)) if gift_nifty.get("DAYCHANGE") else 0
        gift_pct_change = float(gift_nifty.get("PERCHANGE", 0)) if gift_nifty.get("PERCHANGE") else 0
        gift_expiry = gift_nifty.get("EXPIRYDATE", "N/A")
        gift_contracts = int(gift_nifty.get("CONTRACTSTRADED", 0)) if gift_nifty.get("CONTRACTSTRADED") else 0
        
        # Get previous NIFTY close
        nifty_close = float(indicative_nifty.get("closingValue", 0)) if indicative_nifty.get("closingValue") else 0
        nifty_final_close = float(indicative_nifty.get("finalClosingValue", 0)) if indicative_nifty.get("finalClosingValue") else nifty_close
        
        # Gap probability calculation
        if nifty_final_close > 0 and gift_last > 0:
            expected_gap_pct = (gift_last - nifty_final_close) / nifty_final_close * 100
            expected_gap_points = gift_last - nifty_final_close
        else:
            expected_gap_pct = 0
            expected_gap_points = 0
        
        # Overnight sentiment
        if expected_gap_pct > 0.5:
            overnight_sentiment = "Strong Gap Up Expected"
        elif expected_gap_pct > 0.15:
            overnight_sentiment = "Moderate Gap Up Expected"
        elif expected_gap_pct > 0:
            overnight_sentiment = "Flat to Slight Gap Up"
        elif expected_gap_pct > -0.15:
            overnight_sentiment = "Flat to Slight Gap Down"
        elif expected_gap_pct > -0.5:
            overnight_sentiment = "Moderate Gap Down Expected"
        else:
            overnight_sentiment = "Strong Gap Down Expected"
        
        # Global cue strength
        if abs(gift_pct_change) > 0.5:
            global_cue_strength = "Strong" if gift_pct_change > 0 else "Strong Negative"
        elif abs(gift_pct_change) > 0.2:
            global_cue_strength = "Moderate" if gift_pct_change > 0 else "Moderate Negative"
        else:
            global_cue_strength = "Weak/Neutral"
        
        signals["gift_nifty"] = {
            "last_price": gift_last,
            "day_change": gift_change,
            "pct_change": gift_pct_change,
            "expiry": gift_expiry,
            "contracts_traded": gift_contracts,
            "nifty_prev_close": nifty_final_close,
            "expected_gap_pct": round(expected_gap_pct, 2),
            "expected_gap_points": round(expected_gap_points, 2),
            "overnight_sentiment": overnight_sentiment,
            "global_cue_strength": global_cue_strength,
            "directional_bias": "Bullish" if expected_gap_pct > 0.1 else ("Bearish" if expected_gap_pct < -0.1 else "Neutral")
        }
        
        # ========== 2. MARKET REGIME & TRADABILITY ==========
        
        # Parse market states
        market_status_map = {}
        for state in market_states:
            market_name = state.get("market", "Unknown")
            status = state.get("marketStatus", "Unknown")
            market_status_map[market_name] = {
                "status": status,
                "trade_date": state.get("tradeDate", ""),
                "message": state.get("marketStatusMessage", ""),
                "last": state.get("last", ""),
                "variation": state.get("variation", ""),
                "pct_change": state.get("percentChange", "")
            }
        
        # Equity market status
        capital_market = market_status_map.get("Capital Market", {})
        equity_status = capital_market.get("status", "Unknown")
        equity_is_open = equity_status.lower() == "open"
        
        # Currency market
        currency_future = market_status_map.get("currencyfuture", {})
        usdinr_last = float(currency_future.get("last", 0)) if currency_future.get("last") else 0
        usdinr_expiry = currency_future.get("expiryDate", "") if "expiryDate" in currency_future else ""
        
        # Trading regime recommendation
        if equity_is_open:
            trading_regime = "Intraday Active"
            recommendation_mode = "Real-time signals valid"
        else:
            trading_regime = "Pre-market / After-hours"
            recommendation_mode = "Positional signals only - No intraday trades"
        
        signals["market_regime"] = {
            "equity_market_status": equity_status,
            "equity_is_open": equity_is_open,
            "trading_regime": trading_regime,
            "recommendation_mode": recommendation_mode,
            "currency_market_status": market_status_map.get("Currency", {}).get("status", "Unknown"),
            "commodity_market_status": market_status_map.get("Commodity", {}).get("status", "Unknown"),
            "debt_market_status": market_status_map.get("Debt", {}).get("status", "Unknown")
        }
        
        # ========== 3. INDEX-LEVEL RISK TONE ==========
        
        nifty_change = float(indicative_nifty.get("change", 0)) if indicative_nifty.get("change") else 0
        nifty_pct_change = float(indicative_nifty.get("perChange", 0)) if indicative_nifty.get("perChange") else 0
        nifty_status = indicative_nifty.get("status", "N/A")
        
        # Risk tone determination
        if nifty_pct_change > 1:
            risk_tone = "Strong Risk-On"
            confidence_scaling = "Aggressive"
        elif nifty_pct_change > 0.3:
            risk_tone = "Moderate Risk-On"
            confidence_scaling = "Normal Bullish"
        elif nifty_pct_change > -0.3:
            risk_tone = "Neutral"
            confidence_scaling = "Balanced"
        elif nifty_pct_change > -1:
            risk_tone = "Moderate Risk-Off"
            confidence_scaling = "Cautious"
        else:
            risk_tone = "Strong Risk-Off"
            confidence_scaling = "Defensive"
        
        signals["index_risk_tone"] = {
            "nifty_close": nifty_final_close,
            "nifty_change": nifty_change,
            "nifty_pct_change": nifty_pct_change,
            "nifty_status": nifty_status,
            "risk_tone": risk_tone,
            "confidence_scaling": confidence_scaling,
            "broad_market_bias": "Bullish" if nifty_pct_change > 0.1 else ("Bearish" if nifty_pct_change < -0.1 else "Neutral")
        }
        
        # ========== 4. ASSET-CLASS FLOW HINTS ==========
        
        commodity_open = market_status_map.get("Commodity", {}).get("status", "").lower() == "open"
        equity_closed = not equity_is_open
        
        # Cross-asset signals
        if commodity_open and equity_closed:
            commodity_signal = "Commodity markets active - Watch for energy/metal cues"
        else:
            commodity_signal = "Normal trading hours"
        
        # Currency impact analysis
        if usdinr_last > 0:
            if usdinr_last > 85:
                inr_status = "Weak INR"
                sector_impact = "Positive for IT, Pharma exporters. Negative for OMCs, import-heavy sectors"
            elif usdinr_last > 82:
                inr_status = "Moderate INR"
                sector_impact = "Neutral impact on sectors"
            else:
                inr_status = "Strong INR"
                sector_impact = "Positive for OMCs, import-heavy sectors. Negative for exporters"
        else:
            inr_status = "Data unavailable"
            sector_impact = "Unable to assess"
        
        signals["asset_class_flows"] = {
            "commodity_market_open": commodity_open,
            "commodity_signal": commodity_signal,
            "usdinr_rate": usdinr_last,
            "usdinr_expiry": usdinr_expiry,
            "inr_status": inr_status,
            "sector_impact": sector_impact,
            "cross_asset_hint": "Watch commodity moves for next-day sector bias" if commodity_open and equity_closed else "Standard equity focus"
        }
        
        # ========== 5. MACRO BACKDROP ==========
        
        market_cap_tr_usd = float(market_cap.get("marketCapinTRDollars", 0)) if market_cap.get("marketCapinTRDollars") else 0
        market_cap_lac_cr = float(market_cap.get("marketCapinLACCRRupees", 0)) if market_cap.get("marketCapinLACCRRupees") else 0
        market_cap_timestamp = market_cap.get("timeStamp", "N/A")
        
        # Market cap context (India's market cap benchmarks)
        if market_cap_tr_usd > 5:
            cap_phase = "Elevated Valuations"
            liquidity_signal = "High Liquidity - Bull Phase"
        elif market_cap_tr_usd > 4:
            cap_phase = "Healthy Growth"
            liquidity_signal = "Normal Liquidity"
        elif market_cap_tr_usd > 3:
            cap_phase = "Moderate"
            liquidity_signal = "Cautious Liquidity"
        else:
            cap_phase = "Undervalued / Correction"
            liquidity_signal = "Tight Liquidity - Bear Phase"
        
        signals["macro_backdrop"] = {
            "total_market_cap_tr_usd": market_cap_tr_usd,
            "total_market_cap_lac_cr_inr": round(market_cap_lac_cr, 2),
            "market_cap_formatted": market_cap.get("marketCapinLACCRRupeesFormatted", "N/A"),
            "timestamp": market_cap_timestamp,
            "market_phase": cap_phase,
            "liquidity_signal": liquidity_signal,
            "macro_bias": "Bullish" if cap_phase in ["Elevated Valuations", "Healthy Growth"] else ("Bearish" if cap_phase == "Undervalued / Correction" else "Neutral")
        }
        
        # ========== 6. COMPOSITE MARKET STATUS SIGNAL ==========
        
        bullish_count = 0
        bearish_count = 0
        
        if expected_gap_pct > 0.1: bullish_count += 1
        elif expected_gap_pct < -0.1: bearish_count += 1
        
        if gift_pct_change > 0: bullish_count += 1
        elif gift_pct_change < 0: bearish_count += 1
        
        if nifty_pct_change > 0: bullish_count += 1
        elif nifty_pct_change < 0: bearish_count += 1
        
        if "Risk-On" in risk_tone: bullish_count += 1
        elif "Risk-Off" in risk_tone: bearish_count += 1
        
        if cap_phase in ["Elevated Valuations", "Healthy Growth"]: bullish_count += 1
        elif cap_phase == "Undervalued / Correction": bearish_count += 1
        
        if bullish_count > bearish_count + 2:
            market_bias = "Strong Bullish"
        elif bullish_count > bearish_count:
            market_bias = "Bullish"
        elif bearish_count > bullish_count + 2:
            market_bias = "Strong Bearish"
        elif bearish_count > bullish_count:
            market_bias = "Bearish"
        else:
            market_bias = "Neutral"
        
        signals["composite_market_signal"] = {
            "bullish_factors": bullish_count,
            "bearish_factors": bearish_count,
            "market_bias": market_bias,
            "next_session_expectation": overnight_sentiment,
            "tradability": "Active" if equity_is_open else "Pre/Post Market Only"
        }
        
        return signals
    
    # ==================== INDEX INTRADAY SIGNALS (Layer 3) ====================
    
    def extract_index_intraday_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract NIFTY 50 index intraday signals and correlate with equity:
        - Index intraday trend & momentum
        - Volatility regime
        - Key index levels
        - Market structure
        - Index strength score
        - Correlation with equity intraday
        """
        signals = {"available": False}
        
        # Fetch index intraday data
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT json_data FROM layer3_data 
            WHERE data_type = 'index_intraday'
        ''')
        
        row = cursor.fetchone()
        if not row:
            return signals
        
        import json
        data = json.loads(row[0]).get('data', {})
        
        index_graph = data.get('grapthData', [])
        index_name = data.get('identifier', 'NIFTY 50')
        index_close = float(data.get('closePrice', 0)) if data.get('closePrice') else 0
        
        if len(index_graph) < 30:
            return signals
        
        # Fetch equity intraday data for correlation
        intraday_json = self._get_layer1_json(symbol, "intraday_data")
        if not intraday_json:
            return signals
        
        equity_graph = intraday_json.get('grapthData', [])
        
        if len(equity_graph) < 30:
            return signals
        
        signals["available"] = True
        signals["index_name"] = index_name
        
        # Parse index data
        index_prices = np.array([float(p[1]) for p in index_graph])
        index_times = np.array([p[0] for p in index_graph])
        
        # Parse equity data
        equity_prices = np.array([float(p[1]) for p in equity_graph])
        equity_times = np.array([p[0] for p in equity_graph])
        
        signals["data_points"] = {
            "index": len(index_prices),
            "equity": len(equity_prices)
        }
        
        # ========== 1. INDEX INTRADAY TREND & MOMENTUM ==========
        
        index_open = index_prices[0]
        index_high = np.max(index_prices)
        index_low = np.min(index_prices)
        index_current = index_prices[-1]
        index_range = index_high - index_low
        
        # Range position
        if index_range > 0:
            range_position = (index_current - index_low) / index_range * 100
        else:
            range_position = 50
        
        # Session mean (VWAP proxy)
        session_mean = np.mean(index_prices)
        
        # Trend direction
        if index_current > session_mean and index_current > index_open:
            index_trend = "Bullish"
        elif index_current < session_mean and index_current < index_open:
            index_trend = "Bearish"
        else:
            index_trend = "Range-bound"
        
        # Higher highs / lower lows analysis
        n = len(index_prices)
        segment_size = n // 5
        segment_highs = []
        segment_lows = []
        for i in range(5):
            start = i * segment_size
            end = (i + 1) * segment_size if i < 4 else n
            segment_highs.append(np.max(index_prices[start:end]))
            segment_lows.append(np.min(index_prices[start:end]))
        
        higher_highs = sum(1 for i in range(1, len(segment_highs)) if segment_highs[i] > segment_highs[i-1])
        lower_lows = sum(1 for i in range(1, len(segment_lows)) if segment_lows[i] < segment_lows[i-1])
        higher_lows = sum(1 for i in range(1, len(segment_lows)) if segment_lows[i] > segment_lows[i-1])
        lower_highs = sum(1 for i in range(1, len(segment_highs)) if segment_highs[i] < segment_highs[i-1])
        
        # Momentum shift detection
        if higher_highs >= 3 and higher_lows >= 2:
            momentum_shift = "Bullish Momentum (HH + HL)"
        elif lower_lows >= 3 and lower_highs >= 2:
            momentum_shift = "Bearish Momentum (LL + LH)"
        elif higher_highs >= 2 and lower_lows >= 2:
            momentum_shift = "Choppy / No Clear Momentum"
        else:
            momentum_shift = "Transitional"
        
        signals["index_trend_momentum"] = {
            "open": round(index_open, 2),
            "high": round(index_high, 2),
            "low": round(index_low, 2),
            "current": round(index_current, 2),
            "change_pts": round(index_current - index_open, 2),
            "change_pct": round((index_current - index_open) / index_open * 100, 2) if index_open > 0 else 0,
            "session_mean": round(session_mean, 2),
            "price_vs_mean": "Above" if index_current > session_mean else "Below",
            "range_position_pct": round(range_position, 2),
            "trend": index_trend,
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "momentum_shift": momentum_shift
        }
        
        # ========== 2. VOLATILITY REGIME (Intraday) ==========
        
        # Intraday range analysis
        intraday_range_pct = (index_range / index_open) * 100 if index_open > 0 else 0
        
        # Calculate returns for volatility
        returns = np.diff(index_prices) / index_prices[:-1]
        
        # Split into halves
        mid = len(returns) // 2
        first_half_vol = np.std(returns[:mid]) * 100 if mid > 0 else 0
        second_half_vol = np.std(returns[mid:]) * 100 if mid > 0 else 0
        
        # Volatility regime
        if second_half_vol > first_half_vol * 1.5:
            vol_regime = "Expansion"
        elif second_half_vol < first_half_vol * 0.7:
            vol_regime = "Contraction"
        else:
            vol_regime = "Stable"
        
        # Impulse vs consolidation
        # High range + directional = impulse, Low range = consolidation
        if intraday_range_pct > 1 and abs(range_position - 50) > 30:
            phase = "Impulse (Strong Directional Move)"
        elif intraday_range_pct < 0.5:
            phase = "Tight Consolidation"
        else:
            phase = "Normal Trading"
        
        # Trend day vs mean-reversion
        price_changes = np.diff(index_prices)
        positive_moves = np.sum(price_changes > 0)
        negative_moves = np.sum(price_changes < 0)
        
        if abs(positive_moves - negative_moves) > len(price_changes) * 0.3:
            day_type = "Trend Day" if index_current != index_open else "Indecisive Trend"
        else:
            day_type = "Mean-Reversion Day"
        
        signals["volatility_regime"] = {
            "intraday_range_pts": round(index_range, 2),
            "intraday_range_pct": round(intraday_range_pct, 2),
            "first_half_vol": round(first_half_vol, 4),
            "second_half_vol": round(second_half_vol, 4),
            "regime": vol_regime,
            "phase": phase,
            "day_type": day_type
        }
        
        # ========== 3. KEY INDEX LEVELS ==========
        
        # Opening range (first 15-20 mins approx)
        opening_bars = min(20, n // 10)
        opening_range_high = np.max(index_prices[:opening_bars])
        opening_range_low = np.min(index_prices[:opening_bars])
        
        # Current position relative to opening range
        if index_current > opening_range_high:
            or_position = "Above Opening Range (Bullish)"
        elif index_current < opening_range_low:
            or_position = "Below Opening Range (Bearish)"
        else:
            or_position = "Within Opening Range"
        
        # Equilibrium zone (session mean ± std)
        price_std = np.std(index_prices)
        equilibrium_high = session_mean + price_std * 0.5
        equilibrium_low = session_mean - price_std * 0.5
        
        # Support / Resistance from price clusters
        # Use histogram to find price levels with most time spent
        hist, bins = np.histogram(index_prices, bins=20)
        top_levels_idx = np.argsort(hist)[-3:]  # Top 3 price levels
        key_levels = sorted([(bins[i] + bins[i+1]) / 2 for i in top_levels_idx])
        
        signals["key_levels"] = {
            "opening_range_high": round(opening_range_high, 2),
            "opening_range_low": round(opening_range_low, 2),
            "opening_range_position": or_position,
            "equilibrium_high": round(equilibrium_high, 2),
            "equilibrium_low": round(equilibrium_low, 2),
            "current_vs_equilibrium": "Above" if index_current > equilibrium_high else ("Below" if index_current < equilibrium_low else "Within"),
            "intraday_support": round(key_levels[0], 2) if key_levels else round(index_low, 2),
            "intraday_resistance": round(key_levels[-1], 2) if key_levels else round(index_high, 2)
        }
        
        # ========== 4. MARKET STRUCTURE ==========
        
        # Trend persistence: how often price moves in same direction
        same_direction = sum(1 for i in range(1, len(price_changes)) if price_changes[i] * price_changes[i-1] > 0)
        persistence_ratio = same_direction / (len(price_changes) - 1) if len(price_changes) > 1 else 0.5
        
        if persistence_ratio > 0.6:
            structure = "Trending (High Persistence)"
        elif persistence_ratio < 0.4:
            structure = "Choppy (Low Persistence)"
        else:
            structure = "Mixed Structure"
        
        # Breakout detection
        if index_current > opening_range_high * 1.002:
            breakout_status = "Bullish Breakout"
        elif index_current < opening_range_low * 0.998:
            breakout_status = "Bearish Breakdown"
        else:
            breakout_status = "No Breakout"
        
        # Fake breakout check (if broke but returned)
        crossed_above = any(p > opening_range_high for p in index_prices)
        crossed_below = any(p < opening_range_low for p in index_prices)
        
        if crossed_above and index_current < opening_range_high:
            fake_breakout = "Failed Bullish Breakout"
        elif crossed_below and index_current > opening_range_low:
            fake_breakout = "Failed Bearish Breakdown"
        else:
            fake_breakout = "No False Breakout"
        
        # Afternoon reversal probability
        # Split into morning (first 50%) and afternoon (last 50%)
        morning_trend = index_prices[n//2] - index_prices[0]
        afternoon_trend = index_prices[-1] - index_prices[n//2]
        
        if morning_trend * afternoon_trend < 0:
            reversal_detected = "Reversal Detected"
            reversal_strength = abs(afternoon_trend / morning_trend) if morning_trend != 0 else 0
        else:
            reversal_detected = "No Reversal"
            reversal_strength = 0
        
        signals["market_structure"] = {
            "persistence_ratio": round(persistence_ratio, 2),
            "structure": structure,
            "breakout_status": breakout_status,
            "fake_breakout": fake_breakout,
            "reversal_detected": reversal_detected,
            "reversal_strength": round(reversal_strength, 2)
        }
        
        # ========== 5. INDEX STRENGTH SCORE ==========
        
        net_change = index_current - index_open
        net_change_pct = (net_change / index_open) * 100 if index_open > 0 else 0
        
        # Time above session midpoint
        midpoint = (index_high + index_low) / 2
        time_above_mid = np.sum(index_prices > midpoint) / len(index_prices) * 100
        
        # Trend efficiency: net move / total movement
        total_movement = np.sum(np.abs(price_changes))
        trend_efficiency = abs(net_change) / total_movement if total_movement > 0 else 0
        
        # Composite strength score (0-100)
        strength_score = (
            (range_position / 100 * 30) +  # Position in range (30%)
            (time_above_mid / 100 * 30) +  # Time above midpoint (30%)
            (trend_efficiency * 40)  # Trend efficiency (40%)
        )
        
        if net_change < 0:
            strength_score = 100 - strength_score  # Invert for down days
        
        signals["index_strength"] = {
            "net_change_pts": round(net_change, 2),
            "net_change_pct": round(net_change_pct, 2),
            "time_above_midpoint_pct": round(time_above_mid, 2),
            "trend_efficiency": round(trend_efficiency, 2),
            "strength_score": round(strength_score, 2),
            "strength_rating": "Strong" if strength_score > 60 else ("Weak" if strength_score < 40 else "Neutral")
        }
        
        # ========== 6. CORRELATION WITH EQUITY ==========
        
        equity_open = equity_prices[0]
        equity_current = equity_prices[-1]
        equity_high = np.max(equity_prices)
        equity_low = np.min(equity_prices)
        
        # Returns for same time window
        index_return = (index_current - index_open) / index_open * 100 if index_open > 0 else 0
        equity_return = (equity_current - equity_open) / equity_open * 100 if equity_open > 0 else 0
        
        # Relative strength
        if index_return != 0:
            relative_strength = equity_return / index_return
        else:
            relative_strength = 1 if equity_return == 0 else (2 if equity_return > 0 else 0)
        
        # Stock vs Index direction
        if equity_return > 0 and index_return < 0:
            direction_signal = "Stock-Specific Strength (Up while Index Down)"
        elif equity_return < 0 and index_return > 0:
            direction_signal = "Stock-Specific Weakness (Down while Index Up)"
        elif equity_return > 0 and index_return > 0:
            direction_signal = "Moving with Index (Both Up)"
        elif equity_return < 0 and index_return < 0:
            direction_signal = "Moving with Index (Both Down)"
        else:
            direction_signal = "Neutral"
        
        signals["relative_strength"] = {
            "index_return_pct": round(index_return, 2),
            "equity_return_pct": round(equity_return, 2),
            "relative_strength_ratio": round(relative_strength, 2),
            "direction_signal": direction_signal,
            "alpha_pct": round(equity_return - index_return, 2)
        }
        
        # ========== 7. BETA CONFIRMATION ==========
        
        # Calculate intraday beta
        min_len = min(len(index_prices), len(equity_prices))
        if min_len >= 20:
            idx_rets = np.diff(index_prices[-min_len:]) / index_prices[-min_len:-1]
            eq_rets = np.diff(equity_prices[-min_len:]) / equity_prices[-min_len:-1]
            
            valid_mask = np.isfinite(idx_rets) & np.isfinite(eq_rets)
            idx_rets = idx_rets[valid_mask]
            eq_rets = eq_rets[valid_mask]
            
            if len(idx_rets) > 10:
                cov = np.cov(eq_rets, idx_rets)[0, 1]
                idx_var = np.var(idx_rets)
                intraday_beta = cov / idx_var if idx_var > 0 else 1
                correlation = np.corrcoef(eq_rets, idx_rets)[0, 1]
            else:
                intraday_beta = 1
                correlation = 0
        else:
            intraday_beta = 1
            correlation = 0
        
        # Beta interpretation
        if intraday_beta > 1.3:
            beta_signal = "High Beta (Amplifies Index Moves)"
        elif intraday_beta < 0.7:
            beta_signal = "Low Beta (Stock-Specific Play)"
        elif abs(correlation) < 0.3:
            beta_signal = "Uncorrelated (Independent Move)"
        else:
            beta_signal = "Normal Beta"
        
        signals["beta_analysis"] = {
            "intraday_beta": round(intraday_beta, 2) if not np.isnan(intraday_beta) else 1.0,
            "correlation": round(correlation, 2) if not np.isnan(correlation) else 0,
            "beta_signal": beta_signal
        }
        
        # ========== 8. SIGNAL VALIDATION & TRADING SIGNALS ==========
        
        # Long validation
        if equity_return > 0.2 and index_trend == "Bullish":
            long_signal = "Valid Long (Stock Up + Index Bullish)"
            signal_quality = "High"
        elif equity_return > 0.2 and index_trend == "Range-bound":
            long_signal = "Caution Long (Stock Up but Index Flat)"
            signal_quality = "Medium"
        elif equity_return > 0.2 and index_trend == "Bearish":
            long_signal = "Stock-Specific Long (Against Index)"
            signal_quality = "Low - High Risk"
        else:
            long_signal = "No Long Signal"
            signal_quality = "N/A"
        
        # Short validation
        if equity_return < -0.2 and index_trend == "Bearish":
            short_signal = "Valid Short (Stock Down + Index Bearish)"
        elif equity_return < -0.2 and index_trend == "Bullish":
            short_signal = "Stock-Specific Short (Against Index)"
        else:
            short_signal = "No Short Signal"
        
        # False breakout filter
        if breakout_status != "No Breakout" and index_trend == "Range-bound":
            false_breakout_risk = "High (Breakout in Range-bound Market)"
        elif breakout_status != "No Breakout" and fake_breakout != "No False Breakout":
            false_breakout_risk = "High (Recent Fake Breakout)"
        elif breakout_status != "No Breakout" and structure == "Choppy (Low Persistence)":
            false_breakout_risk = "Medium (Choppy Structure)"
        else:
            false_breakout_risk = "Low"
        
        signals["trading_signals"] = {
            "long_signal": long_signal,
            "short_signal": short_signal,
            "signal_quality": signal_quality,
            "false_breakout_risk": false_breakout_risk,
            "index_confirmation": index_trend == "Bullish" if equity_return > 0 else (index_trend == "Bearish" if equity_return < 0 else True),
            "trade_recommendation": "Take Trade" if signal_quality in ["High", "Medium"] else "Avoid/Wait"
        }
        
        # ========== 9. COMPOSITE INDEX SIGNAL ==========
        
        bullish_count = 0
        bearish_count = 0
        
        if index_trend == "Bullish": bullish_count += 1
        elif index_trend == "Bearish": bearish_count += 1
        
        if index_current > session_mean: bullish_count += 1
        else: bearish_count += 1
        
        if "Bullish" in momentum_shift: bullish_count += 1
        elif "Bearish" in momentum_shift: bearish_count += 1
        
        if "Above" in or_position: bullish_count += 1
        elif "Below" in or_position: bearish_count += 1
        
        if day_type == "Trend Day" and index_return > 0: bullish_count += 1
        elif day_type == "Trend Day" and index_return < 0: bearish_count += 1
        
        if strength_score > 55: bullish_count += 1
        elif strength_score < 45: bearish_count += 1
        
        if bullish_count > bearish_count + 2:
            index_bias = "Strong Bullish"
        elif bullish_count > bearish_count:
            index_bias = "Bullish"
        elif bearish_count > bullish_count + 2:
            index_bias = "Strong Bearish"
        elif bearish_count > bullish_count:
            index_bias = "Bearish"
        else:
            index_bias = "Neutral"
        
        signals["composite_index_signal"] = {
            "bullish_factors": bullish_count,
            "bearish_factors": bearish_count,
            "index_bias": index_bias,
            "equity_vs_index": direction_signal,
            "trade_with_index": signal_quality in ["High", "Medium"]
        }
        
        return signals
    
    # ==================== FII/DII INSTITUTIONAL FLOW SIGNALS (Layer 3) ====================
    
    def extract_fii_dii_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Extract FII/DII institutional flow signals:
        - Institutional flow bias (net buying/selling)
        - Flow strength
        - FII vs DII divergence
        - Trend persistence
        - Market regime indicator
        - Timing filter
        """
        signals = {"available": False}
        
        # Fetch FII/DII data
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT json_data FROM layer3_data 
            WHERE data_type = 'fii_dii_trade'
        ''')
        
        row = cursor.fetchone()
        if not row:
            return signals
        
        import json
        data = json.loads(row[0])
        
        latest_data = data.get("latest_from_api", [])
        historical_data = data.get("historical_from_db", [])
        
        if not latest_data:
            return signals
        
        signals["available"] = True
        
        # Parse latest data
        fii_today = None
        dii_today = None
        
        for entry in latest_data:
            cat = entry.get("category", "")
            if "FII" in cat or "FPI" in cat:
                fii_today = {
                    "date": entry.get("date"),
                    "buy": float(entry.get("buyValue", 0).replace(",", "")) if entry.get("buyValue") else 0,
                    "sell": float(entry.get("sellValue", 0).replace(",", "")) if entry.get("sellValue") else 0,
                    "net": float(entry.get("netValue", 0).replace(",", "")) if entry.get("netValue") else 0
                }
            elif "DII" in cat:
                dii_today = {
                    "date": entry.get("date"),
                    "buy": float(entry.get("buyValue", 0).replace(",", "")) if entry.get("buyValue") else 0,
                    "sell": float(entry.get("sellValue", 0).replace(",", "")) if entry.get("sellValue") else 0,
                    "net": float(entry.get("netValue", 0).replace(",", "")) if entry.get("netValue") else 0
                }
        
        if not fii_today or not dii_today:
            return signals
        
        signals["trade_date"] = fii_today.get("date", "N/A")
        
        # ========== 1. INSTITUTIONAL FLOW BIAS ==========
        
        fii_net = fii_today["net"]
        dii_net = dii_today["net"]
        total_net = fii_net + dii_net
        
        # FII stance
        if fii_net > 500:
            fii_stance = "Strong Buying"
        elif fii_net > 100:
            fii_stance = "Moderate Buying"
        elif fii_net > -100:
            fii_stance = "Neutral"
        elif fii_net > -500:
            fii_stance = "Moderate Selling"
        else:
            fii_stance = "Strong Selling"
        
        # DII stance
        if dii_net > 500:
            dii_stance = "Strong Buying"
        elif dii_net > 100:
            dii_stance = "Moderate Buying"
        elif dii_net > -100:
            dii_stance = "Neutral"
        elif dii_net > -500:
            dii_stance = "Moderate Selling"
        else:
            dii_stance = "Strong Selling"
        
        # Combined institutional stance
        if total_net > 1000:
            institutional_bias = "Strong Bullish"
        elif total_net > 300:
            institutional_bias = "Bullish"
        elif total_net > -300:
            institutional_bias = "Neutral"
        elif total_net > -1000:
            institutional_bias = "Bearish"
        else:
            institutional_bias = "Strong Bearish"
        
        signals["flow_bias"] = {
            "fii_net_cr": round(fii_net, 2),
            "dii_net_cr": round(dii_net, 2),
            "total_net_cr": round(total_net, 2),
            "fii_stance": fii_stance,
            "dii_stance": dii_stance,
            "institutional_bias": institutional_bias
        }
        
        # ========== 2. FLOW STRENGTH ==========
        
        fii_turnover = fii_today["buy"] + fii_today["sell"]
        dii_turnover = dii_today["buy"] + dii_today["sell"]
        total_turnover = fii_turnover + dii_turnover
        
        # Conviction level based on absolute net flow
        abs_total_net = abs(total_net)
        if abs_total_net > 2000:
            conviction = "Very High Conviction"
        elif abs_total_net > 1000:
            conviction = "High Conviction"
        elif abs_total_net > 500:
            conviction = "Moderate Conviction"
        elif abs_total_net > 200:
            conviction = "Low Conviction"
        else:
            conviction = "No Conviction (Churn Day)"
        
        # Participation level
        if total_turnover > 40000:
            participation = "Very High Participation"
        elif total_turnover > 25000:
            participation = "High Participation"
        elif total_turnover > 15000:
            participation = "Normal Participation"
        else:
            participation = "Low Participation"
        
        signals["flow_strength"] = {
            "fii_turnover_cr": round(fii_turnover, 2),
            "dii_turnover_cr": round(dii_turnover, 2),
            "total_turnover_cr": round(total_turnover, 2),
            "absolute_net_cr": round(abs_total_net, 2),
            "conviction": conviction,
            "participation": participation
        }
        
        # ========== 3. FII vs DII DIVERGENCE ==========
        
        if fii_net < -100 and dii_net > 100:
            divergence = "FII Selling + DII Buying (Domestic Support)"
            divergence_signal = "Bullish - DII absorbing FII selling"
        elif fii_net > 100 and dii_net < -100:
            divergence = "FII Buying + DII Selling (Risk-On Rally)"
            divergence_signal = "Bullish - FII driving rally"
        elif fii_net > 100 and dii_net > 100:
            divergence = "Both Buying (Strong Trend Confirmation)"
            divergence_signal = "Strong Bullish - Institutional consensus"
        elif fii_net < -100 and dii_net < -100:
            divergence = "Both Selling (Risk-Off / Distribution)"
            divergence_signal = "Strong Bearish - Institutional exit"
        else:
            divergence = "Mixed / Neutral Flow"
            divergence_signal = "Neutral - No clear institutional direction"
        
        signals["fii_dii_divergence"] = {
            "pattern": divergence,
            "signal": divergence_signal,
            "fii_to_dii_ratio": round(abs(fii_net) / abs(dii_net), 2) if dii_net != 0 else 0,
            "dominant_player": "FII" if abs(fii_net) > abs(dii_net) else "DII"
        }
        
        # ========== 4. TREND PERSISTENCE (using historical data) ==========
        
        # Parse historical data
        fii_history = []
        dii_history = []
        
        for entry in historical_data:
            cat = entry.get("category", "")
            net = float(entry.get("net_value", 0)) if entry.get("net_value") else 0
            date = entry.get("trade_date", "")
            
            if "FII" in cat or "FPI" in cat:
                fii_history.append({"date": date, "net": net})
            elif "DII" in cat:
                dii_history.append({"date": date, "net": net})
        
        # Sort by date (most recent first is assumed)
        # Calculate rolling sums
        def calc_rolling_sum(history, days):
            if len(history) >= days:
                return sum(h["net"] for h in history[:days])
            return sum(h["net"] for h in history)
        
        fii_3d = calc_rolling_sum(fii_history, 3)
        fii_5d = calc_rolling_sum(fii_history, 5)
        fii_10d = calc_rolling_sum(fii_history, 10)
        
        dii_3d = calc_rolling_sum(dii_history, 3)
        dii_5d = calc_rolling_sum(dii_history, 5)
        dii_10d = calc_rolling_sum(dii_history, 10)
        
        total_3d = fii_3d + dii_3d
        total_5d = fii_5d + dii_5d
        total_10d = fii_10d + dii_10d
        
        # Trend detection
        if total_3d > 0 and total_5d > 0:
            flow_trend = "Sustained Accumulation"
        elif total_3d < 0 and total_5d < 0:
            flow_trend = "Sustained Distribution"
        elif total_3d > 0 and total_5d < 0:
            flow_trend = "Shift to Accumulation"
        elif total_3d < 0 and total_5d > 0:
            flow_trend = "Shift to Distribution"
        else:
            flow_trend = "Choppy / No Clear Trend"
        
        signals["trend_persistence"] = {
            "fii_3d_net_cr": round(fii_3d, 2),
            "fii_5d_net_cr": round(fii_5d, 2),
            "dii_3d_net_cr": round(dii_3d, 2),
            "dii_5d_net_cr": round(dii_5d, 2),
            "total_3d_net_cr": round(total_3d, 2),
            "total_5d_net_cr": round(total_5d, 2),
            "flow_trend": flow_trend,
            "data_points": len(fii_history)
        }
        
        # ========== 5. MARKET REGIME INDICATOR ==========
        
        # Based on flow patterns
        if fii_5d > 1000 and flow_trend == "Sustained Accumulation":
            market_regime = "Trending Up (FII-Driven)"
        elif fii_5d < -1000 and flow_trend == "Sustained Distribution":
            market_regime = "Trending Down (FII-Exit)"
        elif abs(fii_3d) < 300 and abs(dii_3d) < 300:
            market_regime = "Range-Bound (Low Conviction)"
        elif "Shift" in flow_trend:
            market_regime = "Transition Phase"
        else:
            market_regime = "Mixed / Uncertain"
        
        # Flow consistency (same direction days)
        if len(fii_history) >= 3:
            fii_consistent = sum(1 for h in fii_history[:3] if (h["net"] > 0) == (fii_today["net"] > 0))
            dii_consistent = sum(1 for h in dii_history[:3] if (h["net"] > 0) == (dii_today["net"] > 0))
        else:
            fii_consistent = 0
            dii_consistent = 0
        
        signals["market_regime"] = {
            "regime": market_regime,
            "fii_consistency_3d": fii_consistent,
            "dii_consistency_3d": dii_consistent,
            "regime_strength": "Strong" if fii_consistent >= 3 or dii_consistent >= 3 else ("Moderate" if fii_consistent >= 2 or dii_consistent >= 2 else "Weak")
        }
        
        # ========== 6. TIMING FILTER ==========
        
        # Aggressive long avoidance
        if fii_net < -500 and fii_3d < -1000:
            long_filter = "Avoid Aggressive Longs (Strong FII Outflows)"
            timing_signal = "Bearish"
        elif fii_net < -200 and fii_3d < -500:
            long_filter = "Caution on Longs (FII Selling)"
            timing_signal = "Cautious"
        else:
            long_filter = "Longs Acceptable"
            timing_signal = "Neutral"
        
        # Pullback buy opportunity
        if fii_net < 0 and dii_net > abs(fii_net) * 0.5:
            pullback_signal = "Pullback Buy Favorable (DII Absorbing FII Selling)"
        elif dii_net > 500 and fii_net < 0:
            pullback_signal = "Strong DII Support - Consider Dips"
        else:
            pullback_signal = "No Special Pullback Setup"
        
        signals["timing_filter"] = {
            "long_filter": long_filter,
            "timing_signal": timing_signal,
            "pullback_signal": pullback_signal,
            "dii_absorption_pct": round(abs(dii_net / fii_net) * 100, 1) if fii_net < 0 else 0
        }
        
        # ========== 7. COMPOSITE FII/DII SIGNAL ==========
        
        bullish_count = 0
        bearish_count = 0
        
        if fii_net > 100: bullish_count += 1
        elif fii_net < -100: bearish_count += 1
        
        if dii_net > 100: bullish_count += 1
        elif dii_net < -100: bearish_count += 1
        
        if total_net > 300: bullish_count += 1
        elif total_net < -300: bearish_count += 1
        
        if "Accumulation" in flow_trend: bullish_count += 1
        elif "Distribution" in flow_trend: bearish_count += 1
        
        if "Both Buying" in divergence: bullish_count += 1
        elif "Both Selling" in divergence: bearish_count += 1
        
        if "Trending Up" in market_regime: bullish_count += 1
        elif "Trending Down" in market_regime: bearish_count += 1
        
        if bullish_count > bearish_count + 2:
            fii_dii_bias = "Strong Bullish"
        elif bullish_count > bearish_count:
            fii_dii_bias = "Bullish"
        elif bearish_count > bullish_count + 2:
            fii_dii_bias = "Strong Bearish"
        elif bearish_count > bullish_count:
            fii_dii_bias = "Bearish"
        else:
            fii_dii_bias = "Neutral"
        
        signals["composite_fii_dii_signal"] = {
            "bullish_factors": bullish_count,
            "bearish_factors": bearish_count,
            "institutional_bias": fii_dii_bias,
            "key_insight": divergence_signal,
            "actionable": timing_signal != "Bearish"
        }
        
        return signals
    
    # ==================== INDEX FUTURES SIGNALS ====================
    
    def extract_index_futures_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from index_futures_data table (NIFTY 50 futures)"""
        signals = {"available": False}
        
        try:
            cursor = self.conn.cursor()
            
            # Get recent data for nearest expiry (sorted by trade_date DESC)
            cursor.execute("""
                SELECT trade_date, expiry_date, open_price, high_price, low_price, 
                       close_price, volume, open_interest, change_in_oi
                FROM index_futures_data 
                WHERE index_name = 'NIFTY'
                ORDER BY trade_date DESC, expiry_date ASC
            """)
            rows = cursor.fetchall()
            
            if not rows:
                return signals
            
            # Group by trade_date, use nearest expiry
            from collections import defaultdict
            date_data = defaultdict(list)
            for row in rows:
                date_data[row[0]].append({
                    "trade_date": row[0],
                    "expiry_date": row[1],
                    "open": row[2],
                    "high": row[3],
                    "low": row[4],
                    "close": row[5],
                    "volume": row[6],
                    "oi": row[7],
                    "oi_change": row[8]
                })
            
            # Get latest trade dates (use nearest expiry for each date)
            sorted_dates = sorted(date_data.keys(), reverse=True)
            if len(sorted_dates) < 2:
                return signals
            
            today_data = date_data[sorted_dates[0]][0]  # Nearest expiry
            prev_data = date_data[sorted_dates[1]][0]   # Previous day nearest expiry
            
            signals["available"] = True
            signals["trade_date"] = today_data["trade_date"]
            signals["expiry_date"] = today_data["expiry_date"]
            
            # ========== 1. INDEX TREND ==========
            today_close = today_data["close"]
            prev_close = prev_data["close"]
            price_change = today_close - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close else 0
            
            if price_change_pct > 0.5:
                trend = "Bullish"
            elif price_change_pct < -0.5:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            signals["index_trend"] = {
                "close": round(today_close, 2),
                "prev_close": round(prev_close, 2),
                "change": round(price_change, 2),
                "change_pct": round(price_change_pct, 2),
                "trend": trend
            }
            
            # ========== 2. OPEN INTEREST ANALYSIS ==========
            today_oi = today_data["oi"]
            prev_oi = prev_data["oi"]
            oi_change = today_data["oi_change"]
            oi_change_pct = (oi_change / prev_oi) * 100 if prev_oi else 0
            
            # Participation level
            if today_oi > 15000000:
                participation = "Very High"
            elif today_oi > 12000000:
                participation = "High"
            elif today_oi > 8000000:
                participation = "Normal"
            else:
                participation = "Low"
            
            # Conviction from OI change
            if abs(oi_change_pct) > 10:
                conviction = "High Conviction"
            elif abs(oi_change_pct) > 5:
                conviction = "Moderate Conviction"
            else:
                conviction = "Low Conviction"
            
            signals["open_interest"] = {
                "oi": round(today_oi, 0),
                "oi_change": round(oi_change, 0),
                "oi_change_pct": round(oi_change_pct, 2),
                "participation": participation,
                "conviction": conviction
            }
            
            # ========== 3. PRICE + OI INTERPRETATION ==========
            price_up = price_change > 0
            oi_up = oi_change > 0
            
            if price_up and oi_up:
                interpretation = "Long Build-Up"
                signal = "Bullish - Fresh longs added"
                buildup_strength = "Strong" if price_change_pct > 0.5 and oi_change_pct > 5 else "Moderate"
            elif not price_up and oi_up:
                interpretation = "Short Build-Up"
                signal = "Bearish - Fresh shorts added"
                buildup_strength = "Strong" if price_change_pct < -0.5 and oi_change_pct > 5 else "Moderate"
            elif price_up and not oi_up:
                interpretation = "Short Covering"
                signal = "Mildly Bullish - Shorts exiting"
                buildup_strength = "Moderate" if price_change_pct > 0.5 else "Weak"
            else:  # price down, oi down
                interpretation = "Long Unwinding"
                signal = "Mildly Bearish - Longs exiting"
                buildup_strength = "Moderate" if price_change_pct < -0.5 else "Weak"
            
            signals["price_oi_interpretation"] = {
                "interpretation": interpretation,
                "signal": signal,
                "buildup_strength": buildup_strength,
                "price_direction": "Up" if price_up else "Down",
                "oi_direction": "Up" if oi_up else "Down"
            }
            
            # ========== 4. VOLUME ANALYSIS ==========
            today_vol = today_data["volume"]
            
            # Calculate average volume from recent data
            recent_volumes = []
            for i, date in enumerate(sorted_dates[:10]):
                if date_data[date]:
                    recent_volumes.append(date_data[date][0]["volume"])
            
            avg_vol = np.mean(recent_volumes) if recent_volumes else today_vol
            vol_ratio = today_vol / avg_vol if avg_vol else 1
            
            if vol_ratio > 1.5:
                vol_signal = "Volume Spike (Institutional Activity)"
            elif vol_ratio > 1.2:
                vol_signal = "Above Average Volume"
            elif vol_ratio < 0.7:
                vol_signal = "Low Volume (Low Participation)"
            else:
                vol_signal = "Normal Volume"
            
            signals["volume_analysis"] = {
                "volume": round(today_vol, 0),
                "avg_volume": round(avg_vol, 0),
                "volume_ratio": round(vol_ratio, 2),
                "volume_signal": vol_signal
            }
            
            # ========== 5. BASIS / PREMIUM ANALYSIS ==========
            # Get spot price from index_intraday if available
            spot_price = None
            try:
                cursor.execute("""
                    SELECT data FROM layer3_data WHERE data_type = 'index_intraday'
                """)
                spot_row = cursor.fetchone()
                if spot_row:
                    import json
                    spot_data = json.loads(spot_row[0])
                    if isinstance(spot_data, dict) and "grapthData" in spot_data:
                        graph = spot_data["grapthData"]
                        if graph:
                            spot_price = graph[-1].get("y")
            except:
                pass
            
            if spot_price:
                basis = today_close - spot_price
                basis_pct = (basis / spot_price) * 100
                
                if basis_pct > 0.3:
                    premium_signal = "High Premium (Risk Appetite High)"
                elif basis_pct > 0.1:
                    premium_signal = "Normal Premium"
                elif basis_pct > -0.1:
                    premium_signal = "At Fair Value"
                elif basis_pct > -0.3:
                    premium_signal = "Slight Discount"
                else:
                    premium_signal = "High Discount (Risk Aversion)"
                
                signals["basis_premium"] = {
                    "futures_price": round(today_close, 2),
                    "spot_price": round(spot_price, 2),
                    "basis": round(basis, 2),
                    "basis_pct": round(basis_pct, 3),
                    "premium_signal": premium_signal
                }
            else:
                signals["basis_premium"] = {
                    "available": False,
                    "note": "Spot price not available"
                }
            
            # ========== 6. EQUITY CORRELATION SIGNALS ==========
            
            # Get equity data for comparison
            equity_signals = {}
            try:
                # Get stock intraday change
                intraday = self.extract_intraday_signals(symbol)
                if intraday.get("available"):
                    stock_change = intraday.get("price_action", {}).get("day_change_pct", 0)
                    
                    # Market regime filter
                    if "Long Build-Up" in interpretation or "Short Covering" in interpretation:
                        market_filter = "Bullish - Favor long equity setups"
                    elif "Short Build-Up" in interpretation or "Long Unwinding" in interpretation:
                        market_filter = "Bearish - Avoid longs, favor hedged trades"
                    else:
                        market_filter = "Neutral"
                    
                    # Stock vs Index strength
                    if stock_change > 0 and price_change_pct < 0:
                        relative_strength = "Outperformer (Stock up, Index down)"
                        rs_signal = "Strong relative strength"
                    elif stock_change < 0 and price_change_pct > 0:
                        relative_strength = "Underperformer (Stock down, Index up)"
                        rs_signal = "Weak relative strength"
                    elif stock_change > price_change_pct:
                        relative_strength = "Mild Outperformer"
                        rs_signal = "Stock leading index"
                    elif stock_change < price_change_pct:
                        relative_strength = "Mild Underperformer"
                        rs_signal = "Stock lagging index"
                    else:
                        relative_strength = "Inline with Index"
                        rs_signal = "Moving with market"
                    
                    equity_signals["market_filter"] = market_filter
                    equity_signals["relative_strength"] = relative_strength
                    equity_signals["rs_signal"] = rs_signal
                    equity_signals["stock_change_pct"] = round(stock_change, 2)
                    equity_signals["index_change_pct"] = round(price_change_pct, 2)
                    equity_signals["alpha"] = round(stock_change - price_change_pct, 2)
            except:
                pass
            
            # Confirmation layer
            if equity_signals:
                if "Long Build-Up" in interpretation and equity_signals.get("stock_change_pct", 0) > 0:
                    equity_signals["confirmation"] = "High Confidence - Stock bullish + Index long build-up"
                    equity_signals["position_size"] = "Full size"
                elif "Short Build-Up" in interpretation and equity_signals.get("stock_change_pct", 0) > 0:
                    equity_signals["confirmation"] = "Low Confidence - Stock bullish but Index bearish"
                    equity_signals["position_size"] = "Reduced size"
                elif "Long Build-Up" in interpretation and equity_signals.get("stock_change_pct", 0) < 0:
                    equity_signals["confirmation"] = "Potential Reversal - Stock weak but Index bullish"
                    equity_signals["position_size"] = "Wait for confirmation"
                else:
                    equity_signals["confirmation"] = "Standard Setup"
                    equity_signals["position_size"] = "Normal size"
            
            signals["equity_correlation"] = equity_signals if equity_signals else {"available": False}
            
            # ========== 7. SECTOR AMPLIFICATION ==========
            sector_amp = {}
            try:
                sector_signals = self.extract_intraday_sector_signals(symbol)
                if sector_signals.get("available"):
                    sector_trend = sector_signals.get("intraday_trend", {}).get("bias", "")
                    
                    if "Bullish" in interpretation and "Bullish" in sector_trend:
                        sector_amp["amplification"] = "Momentum Expansion Expected"
                        sector_amp["signal"] = "Strong index + Strong sector = High breakout success rate"
                    elif "Bearish" in interpretation:
                        sector_amp["amplification"] = "Breakout Caution"
                        sector_amp["signal"] = "Weak index futures = Higher breakout failure rate"
                    else:
                        sector_amp["amplification"] = "Normal Conditions"
                        sector_amp["signal"] = "Standard market behavior"
            except:
                pass
            
            signals["sector_amplification"] = sector_amp if sector_amp else {"available": False}
            
            # ========== 8. COMPOSITE INDEX FUTURES SIGNAL ==========
            bullish_count = 0
            bearish_count = 0
            
            if price_change_pct > 0.3: bullish_count += 1
            elif price_change_pct < -0.3: bearish_count += 1
            
            if "Long Build-Up" in interpretation: bullish_count += 1
            elif "Short Build-Up" in interpretation: bearish_count += 1
            
            if "Short Covering" in interpretation: bullish_count += 0.5
            elif "Long Unwinding" in interpretation: bearish_count += 0.5
            
            if vol_ratio > 1.2 and price_change_pct > 0: bullish_count += 1
            elif vol_ratio > 1.2 and price_change_pct < 0: bearish_count += 1
            
            if signals.get("basis_premium", {}).get("basis_pct", 0) > 0.1: bullish_count += 0.5
            elif signals.get("basis_premium", {}).get("basis_pct", 0) < -0.1: bearish_count += 0.5
            
            total = bullish_count + bearish_count
            if bullish_count > bearish_count + 1:
                futures_bias = "Strong Bullish"
            elif bullish_count > bearish_count:
                futures_bias = "Bullish"
            elif bearish_count > bullish_count + 1:
                futures_bias = "Strong Bearish"
            elif bearish_count > bullish_count:
                futures_bias = "Bearish"
            else:
                futures_bias = "Neutral"
            
            signals["composite_index_futures"] = {
                "bullish_factors": round(bullish_count, 1),
                "bearish_factors": round(bearish_count, 1),
                "futures_bias": futures_bias,
                "key_signal": interpretation,
                "trade_recommendation": "Favor longs" if futures_bias in ["Strong Bullish", "Bullish"] else ("Favor shorts/hedges" if futures_bias in ["Strong Bearish", "Bearish"] else "Neutral stance")
            }
            
            return signals
            
        except Exception as e:
            signals["error"] = str(e)
            return signals
    
    # ==================== INDEX HISTORY SIGNALS ====================
    
    def extract_index_history_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from index_history table (NIFTY 50 historical data)"""
        signals = {"available": False}
        
        try:
            cursor = self.conn.cursor()
            
            # Get all NIFTY 50 data ordered by date
            cursor.execute("""
                SELECT trade_date, open_value, high_value, low_value, close_value, volume, turnover
                FROM index_history 
                WHERE index_name = 'NIFTY 50'
                ORDER BY id DESC
            """)
            rows = cursor.fetchall()
            
            if len(rows) < 20:
                return signals
            
            # Convert to arrays
            dates = [r[0] for r in rows]
            opens = np.array([r[1] for r in rows])
            highs = np.array([r[2] for r in rows])
            lows = np.array([r[3] for r in rows])
            closes = np.array([r[4] for r in rows])
            volumes = np.array([r[5] for r in rows])
            turnovers = np.array([r[6] for r in rows])
            
            signals["available"] = True
            signals["data_points"] = len(rows)
            signals["latest_date"] = dates[0]
            signals["latest_close"] = round(closes[0], 2)
            
            # ========== 1. PRIMARY TREND (MA-based) ==========
            # Data is newest-first, so we calculate simple moving averages directly
            # SMA is just the mean of the last N values
            sma_10_val = float(np.mean(closes[:10])) if len(closes) >= 10 else float(closes[0])
            sma_20_val = float(np.mean(closes[:20])) if len(closes) >= 20 else float(np.mean(closes))
            sma_50_val = float(np.mean(closes[:50])) if len(closes) >= 50 else float(np.mean(closes))
            
            # For slope calculation, we need SMA from 5 days ago
            sma_10_5d_ago = float(np.mean(closes[5:15])) if len(closes) >= 15 else sma_10_val
            
            # Trend determination (current price vs moving averages)
            short_trend = "Bullish" if closes[0] > sma_10_val else "Bearish"
            medium_trend = "Bullish" if closes[0] > sma_20_val else "Bearish"
            
            # MA slope (10-period slope over last 5 days)
            if sma_10_5d_ago > 0:
                ma_slope = (sma_10_val - sma_10_5d_ago) / sma_10_5d_ago * 100
                slope_signal = "Rising" if ma_slope > 0.5 else ("Falling" if ma_slope < -0.5 else "Flat")
            else:
                ma_slope = 0
                slope_signal = "N/A"
            
            # HH-HL / LH-LL structure
            recent_highs = highs[:10]
            recent_lows = lows[:10]
            hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i-1] > recent_highs[i])
            hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i-1] > recent_lows[i])
            
            if hh_count >= 3 and hl_count >= 3:
                structure = "Higher Highs & Higher Lows (Uptrend)"
                structure_bias = "Bullish"
            elif hh_count <= 1 and hl_count <= 1:
                structure = "Lower Highs & Lower Lows (Downtrend)"
                structure_bias = "Bearish"
            else:
                structure = "Mixed Structure (Range/Transition)"
                structure_bias = "Neutral"
            
            signals["primary_trend"] = {
                "short_term": short_trend,
                "medium_term": medium_trend,
                "sma_10": round(sma_10_val, 2),
                "sma_20": round(sma_20_val, 2),
                "sma_50": round(sma_50_val, 2),
                "ma_slope_pct": round(ma_slope, 2),
                "slope_signal": slope_signal,
                "price_structure": structure,
                "structure_bias": structure_bias
            }
            
            # ========== 2. MOMENTUM STRENGTH ==========
            # RSI calculation
            def calc_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return 50
                deltas = np.diff(prices[::-1])  # Reverse for oldest-first
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[:period])
                avg_loss = np.mean(losses[:period])
                if avg_loss == 0:
                    return 100
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))
            
            rsi = calc_rsi(closes)
            
            # Rate of Change
            if len(closes) >= 10:
                roc_10 = (closes[0] - closes[9]) / closes[9] * 100
            else:
                roc_10 = 0
            
            if len(closes) >= 20:
                roc_20 = (closes[0] - closes[19]) / closes[19] * 100
            else:
                roc_20 = 0
            
            # Trend acceleration (comparing recent ROC to older ROC)
            if len(closes) >= 20:
                roc_recent = (closes[0] - closes[4]) / closes[4] * 100
                roc_older = (closes[5] - closes[9]) / closes[9] * 100
                acceleration = roc_recent - roc_older
                if acceleration > 1:
                    accel_signal = "Accelerating Up"
                elif acceleration < -1:
                    accel_signal = "Accelerating Down"
                else:
                    accel_signal = "Steady Momentum"
            else:
                acceleration = 0
                accel_signal = "N/A"
            
            # RSI interpretation
            if rsi > 70:
                rsi_signal = "Overbought"
            elif rsi > 60:
                rsi_signal = "Bullish Momentum"
            elif rsi < 30:
                rsi_signal = "Oversold"
            elif rsi < 40:
                rsi_signal = "Bearish Momentum"
            else:
                rsi_signal = "Neutral"
            
            signals["momentum_strength"] = {
                "rsi": round(rsi, 1),
                "rsi_signal": rsi_signal,
                "roc_10d_pct": round(roc_10, 2),
                "roc_20d_pct": round(roc_20, 2),
                "acceleration": round(acceleration, 2),
                "acceleration_signal": accel_signal
            }
            
            # ========== 3. VOLATILITY REGIME ==========
            # ATR calculation
            def calc_atr(highs, lows, closes, period=14):
                if len(highs) < period + 1:
                    return 0
                tr_list = []
                for i in range(len(highs) - 1):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i+1]),
                        abs(lows[i] - closes[i+1])
                    )
                    tr_list.append(tr)
                if len(tr_list) >= period:
                    return np.mean(tr_list[:period])
                return np.mean(tr_list) if tr_list else 0
            
            atr = calc_atr(highs, lows, closes)
            atr_pct = (atr / closes[0]) * 100 if closes[0] else 0
            
            # Range expansion/contraction (compare recent ATR to older)
            if len(highs) >= 30:
                recent_ranges = highs[:10] - lows[:10]
                older_ranges = highs[10:20] - lows[10:20]
                avg_recent = np.mean(recent_ranges)
                avg_older = np.mean(older_ranges)
                range_ratio = avg_recent / avg_older if avg_older else 1
                
                if range_ratio > 1.3:
                    vol_regime = "Range Expansion (Trending)"
                elif range_ratio < 0.7:
                    vol_regime = "Range Contraction (Consolidating)"
                else:
                    vol_regime = "Normal Volatility"
            else:
                range_ratio = 1
                vol_regime = "N/A"
            
            # Volatility level
            if atr_pct > 2.0:
                vol_level = "High Volatility"
            elif atr_pct > 1.2:
                vol_level = "Moderate Volatility"
            else:
                vol_level = "Low Volatility"
            
            signals["volatility_regime"] = {
                "atr": round(atr, 2),
                "atr_pct": round(atr_pct, 2),
                "volatility_level": vol_level,
                "range_ratio": round(range_ratio, 2),
                "regime": vol_regime
            }
            
            # ========== 4. VOLUME CONFIRMATION ==========
            avg_vol_20 = np.mean(volumes[:20]) if len(volumes) >= 20 else np.mean(volumes)
            vol_ratio = volumes[0] / avg_vol_20 if avg_vol_20 else 1
            
            price_change = closes[0] - closes[1] if len(closes) > 1 else 0
            price_up = price_change > 0
            vol_up = vol_ratio > 1.1
            
            if price_up and vol_up:
                vol_confirm = "Bullish Confirmation (Price Up + Volume Up)"
                conviction = "High"
            elif not price_up and vol_up:
                vol_confirm = "Bearish Confirmation (Price Down + Volume Up)"
                conviction = "High"
            elif price_up and not vol_up:
                vol_confirm = "Weak Rally (Price Up + Low Volume)"
                conviction = "Low"
            else:
                vol_confirm = "Weak Decline (Price Down + Low Volume)"
                conviction = "Low"
            
            signals["volume_confirmation"] = {
                "volume": round(volumes[0], 0),
                "avg_volume_20d": round(avg_vol_20, 0),
                "volume_ratio": round(vol_ratio, 2),
                "confirmation": vol_confirm,
                "conviction": conviction
            }
            
            # ========== 5. DISTRIBUTION / ACCUMULATION ==========
            # Flat price + high volume = distribution/accumulation
            price_range_pct = (highs[0] - lows[0]) / closes[0] * 100 if closes[0] else 0
            is_flat = price_range_pct < 1.0
            is_high_vol = vol_ratio > 1.3
            
            if is_flat and is_high_vol and closes[0] > closes[1]:
                accum_dist = "Accumulation (Flat price + High volume + Slight up)"
            elif is_flat and is_high_vol and closes[0] < closes[1]:
                accum_dist = "Distribution (Flat price + High volume + Slight down)"
            elif is_high_vol:
                accum_dist = "High Activity Day"
            else:
                accum_dist = "Normal Trading"
            
            signals["accumulation_distribution"] = {
                "daily_range_pct": round(price_range_pct, 2),
                "is_narrow_range": is_flat,
                "is_high_volume": is_high_vol,
                "signal": accum_dist
            }
            
            # ========== 6. MARKET BREADTH PROXY ==========
            # Sustained index move implies broad participation
            if len(closes) >= 6:
                up_days = sum(1 for i in range(5) if closes[i] > closes[i+1])
                down_days = 5 - up_days
                
                if up_days >= 4:
                    breadth = "Broad Rally (4+ up days in 5)"
                    breadth_signal = "Strong Bullish"
                elif down_days >= 4:
                    breadth = "Broad Decline (4+ down days in 5)"
                    breadth_signal = "Strong Bearish"
                elif up_days >= 3:
                    breadth = "Mildly Bullish Breadth"
                    breadth_signal = "Bullish"
                elif down_days >= 3:
                    breadth = "Mildly Bearish Breadth"
                    breadth_signal = "Bearish"
                else:
                    breadth = "Mixed Breadth"
                    breadth_signal = "Neutral"
            else:
                breadth = "Insufficient Data"
                breadth_signal = "N/A"
            
            signals["market_breadth_proxy"] = {
                "up_days_5d": up_days if len(closes) >= 6 else 0,
                "down_days_5d": down_days if len(closes) >= 6 else 0,
                "breadth": breadth,
                "signal": breadth_signal
            }
            
            # ========== 7. SUPPORT / RESISTANCE ZONES ==========
            # Key institutional levels (recent swing highs/lows)
            recent_high = np.max(highs[:20])
            recent_low = np.min(lows[:20])
            
            # Pivot levels
            pivot = (highs[0] + lows[0] + closes[0]) / 3
            r1 = 2 * pivot - lows[0]
            s1 = 2 * pivot - highs[0]
            
            # Distance from key levels
            dist_from_high = (recent_high - closes[0]) / closes[0] * 100
            dist_from_low = (closes[0] - recent_low) / closes[0] * 100
            
            if dist_from_high < 1:
                level_signal = "Near Resistance (Breakout Zone)"
            elif dist_from_low < 1:
                level_signal = "Near Support (Bounce Zone)"
            elif dist_from_high < 3:
                level_signal = "Approaching Resistance"
            elif dist_from_low < 3:
                level_signal = "Approaching Support"
            else:
                level_signal = "Mid-Range"
            
            signals["support_resistance"] = {
                "recent_high_20d": round(recent_high, 2),
                "recent_low_20d": round(recent_low, 2),
                "pivot": round(pivot, 2),
                "resistance_r1": round(r1, 2),
                "support_s1": round(s1, 2),
                "dist_from_high_pct": round(dist_from_high, 2),
                "dist_from_low_pct": round(dist_from_low, 2),
                "level_signal": level_signal
            }
            
            # ========== 8. EQUITY CORRELATION ==========
            equity_corr = {}
            try:
                hist = self.extract_historical_signals(symbol)
                if hist.get("available"):
                    stock_return = hist.get("trend_structure", {}).get("percent_change_20d", 0)
                    index_return = roc_20
                    
                    # Relative strength
                    alpha = stock_return - index_return
                    if alpha > 3:
                        rs_signal = "Strong Outperformer (Leader)"
                    elif alpha > 1:
                        rs_signal = "Mild Outperformer"
                    elif alpha < -3:
                        rs_signal = "Strong Underperformer (Laggard)"
                    elif alpha < -1:
                        rs_signal = "Mild Underperformer"
                    else:
                        rs_signal = "Inline with Market"
                    
                    equity_corr["stock_return_20d"] = round(stock_return, 2)
                    equity_corr["index_return_20d"] = round(index_return, 2)
                    equity_corr["alpha_20d"] = round(alpha, 2)
                    equity_corr["relative_strength"] = rs_signal
                    
                    # Signal validation
                    stock_trend = hist.get("trend_structure", {}).get("trend", "")
                    if "Uptrend" in stock_trend and short_trend == "Bullish":
                        equity_corr["signal_validation"] = "High Probability (Both Bullish)"
                    elif "Downtrend" in stock_trend and short_trend == "Bearish":
                        equity_corr["signal_validation"] = "High Probability (Both Bearish)"
                    elif "Uptrend" in stock_trend and short_trend == "Bearish":
                        equity_corr["signal_validation"] = "Caution - Stock bullish vs Index bearish"
                    elif "Downtrend" in stock_trend and short_trend == "Bullish":
                        equity_corr["signal_validation"] = "Caution - Stock bearish vs Index bullish"
                    else:
                        equity_corr["signal_validation"] = "Mixed signals"
                    
                    # Volatility alignment
                    stock_vol = hist.get("volatility", {}).get("volatility_regime", "")
                    if "High" in vol_level:
                        equity_corr["volatility_guidance"] = "Wide stops, lower position size recommended"
                    elif "Low" in vol_level:
                        equity_corr["volatility_guidance"] = "Trend-following strategies favored"
                    else:
                        equity_corr["volatility_guidance"] = "Normal position sizing"
                    
                    # Volume behavior
                    stock_vol_ratio = hist.get("volume_trend", {}).get("volume_ratio", 1)
                    if stock_vol_ratio > 1.5 and vol_ratio < 1.1:
                        equity_corr["volume_signal"] = "Stock-Specific Move (Stock vol high, Index vol normal)"
                    elif stock_vol_ratio > 1.3 and vol_ratio > 1.3:
                        equity_corr["volume_signal"] = "Market-Wide Move (Both high volume)"
                    else:
                        equity_corr["volume_signal"] = "Normal volume pattern"
            except:
                pass
            
            signals["equity_correlation"] = equity_corr if equity_corr else {"available": False}
            
            # ========== 9. COMPOSITE INDEX HISTORY SIGNAL ==========
            bullish_count = 0
            bearish_count = 0
            
            if short_trend == "Bullish": bullish_count += 1
            else: bearish_count += 1
            
            if medium_trend == "Bullish": bullish_count += 1
            else: bearish_count += 1
            
            if "Uptrend" in structure: bullish_count += 1
            elif "Downtrend" in structure: bearish_count += 1
            
            if rsi > 55: bullish_count += 1
            elif rsi < 45: bearish_count += 1
            
            if roc_10 > 1: bullish_count += 1
            elif roc_10 < -1: bearish_count += 1
            
            if "Expansion" in vol_regime: bullish_count += 0.5  # Neutral for volatility
            
            if "Bullish Confirmation" in vol_confirm: bullish_count += 1
            elif "Bearish Confirmation" in vol_confirm: bearish_count += 1
            
            if "Accumulation" in accum_dist: bullish_count += 1
            elif "Distribution" in accum_dist: bearish_count += 1
            
            if "Bullish" in breadth_signal: bullish_count += 1
            elif "Bearish" in breadth_signal: bearish_count += 1
            
            total = bullish_count + bearish_count
            if bullish_count > bearish_count + 2:
                index_bias = "Strong Bullish"
            elif bullish_count > bearish_count:
                index_bias = "Bullish"
            elif bearish_count > bullish_count + 2:
                index_bias = "Strong Bearish"
            elif bearish_count > bullish_count:
                index_bias = "Bearish"
            else:
                index_bias = "Neutral"
            
            signals["composite_index_history"] = {
                "bullish_factors": round(bullish_count, 1),
                "bearish_factors": round(bearish_count, 1),
                "index_bias": index_bias,
                "key_signal": f"{short_trend} trend, {rsi_signal}, {vol_regime}",
                "positional_recommendation": "Favor longs" if "Bullish" in index_bias else ("Favor shorts" if "Bearish" in index_bias else "Range trading")
            }
            
            return signals
            
        except Exception as e:
            signals["error"] = str(e)
            return signals
    
    # ==================== INDEX OPTION CHAIN SIGNALS ====================
    
    def extract_index_option_chain_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from index_option_chain table (NIFTY 50 options)"""
        signals = {"available": False}
        
        try:
            cursor = self.conn.cursor()
            
            # Get ALL expiry dates
            cursor.execute("""
                SELECT DISTINCT expiry_date FROM index_option_chain 
                WHERE index_name = 'NIFTY'
            """)
            raw_expiries = [row[0] for row in cursor.fetchall()]
            
            if not raw_expiries:
                return signals
            
            # Parse and sort expiries chronologically
            from datetime import datetime
            def parse_expiry(exp_str):
                try:
                    return datetime.strptime(exp_str, "%d-%b-%Y")
                except:
                    try:
                        return datetime.strptime(exp_str, "%d-%m-%Y")
                    except:
                        return datetime.max
            
            # Sort by actual date (nearest first)
            all_expiries = sorted(raw_expiries, key=parse_expiry)
            
            # Filter to get weekly/monthly expiries (exclude far-dated LEAPS > 90 days)
            today = datetime.now()
            near_term_expiries = [e for e in all_expiries if (parse_expiry(e) - today).days <= 90]
            
            # Keep first 4 near-term expiries for analysis
            active_expiries = near_term_expiries[:4] if len(near_term_expiries) >= 4 else near_term_expiries
            if not active_expiries:
                active_expiries = all_expiries[:4]  # Fallback
            
            nearest_expiry = active_expiries[0]
            
            # Helper function to load option data for an expiry
            def load_expiry_data(expiry):
                cursor.execute("""
                    SELECT strike_price, option_type, open_interest, change_in_oi, 
                           volume, iv, ltp, bid_price, ask_price
                    FROM index_option_chain 
                    WHERE index_name = 'NIFTY' AND expiry_date = ?
                    ORDER BY strike_price
                """, (expiry,))
                rows = cursor.fetchall()
                
                calls_data = {}
                puts_data = {}
                strikes_set = set()
                
                for row in rows:
                    strike = row[0]
                    opt_type = row[1]
                    strikes_set.add(strike)
                    
                    data = {
                        "oi": row[2] or 0,
                        "chg_oi": row[3] or 0,
                        "volume": row[4] or 0,
                        "iv": row[5] or 0,
                        "ltp": row[6] or 0,
                        "bid": row[7] or 0,
                        "ask": row[8] or 0
                    }
                    
                    if opt_type == "CE":
                        calls_data[strike] = data
                    else:
                        puts_data[strike] = data
                
                return calls_data, puts_data, sorted(strikes_set)
            
            # Load data for all active expiries
            expiry_data = {}
            for exp in active_expiries:
                calls_exp, puts_exp, strikes_exp = load_expiry_data(exp)
                if calls_exp or puts_exp:
                    expiry_data[exp] = {
                        "calls": calls_exp,
                        "puts": puts_exp,
                        "strikes": strikes_exp,
                        "total_call_oi": sum(c["oi"] for c in calls_exp.values()),
                        "total_put_oi": sum(p["oi"] for p in puts_exp.values())
                    }
            
            if not expiry_data:
                return signals
            
            # Primary analysis uses nearest expiry
            calls = expiry_data[nearest_expiry]["calls"]
            puts = expiry_data[nearest_expiry]["puts"]
            strikes = expiry_data[nearest_expiry]["strikes"]
            
            signals["available"] = True
            signals["primary_expiry"] = nearest_expiry
            signals["all_expiries"] = active_expiries
            signals["total_strikes"] = len(strikes)
            
            # Calculate totals across all expiries for broader view
            total_call_oi_all = sum(e["total_call_oi"] for e in expiry_data.values())
            total_put_oi_all = sum(e["total_put_oi"] for e in expiry_data.values())
            
            # Primary expiry totals
            total_call_oi = expiry_data[nearest_expiry]["total_call_oi"]
            total_put_oi = expiry_data[nearest_expiry]["total_put_oi"]
            
            # Find ATM strike (use mid-strike)
            mid_strike = strikes[len(strikes)//2] if strikes else 0
            
            # ========== 1. MAX PAIN CALCULATION ==========
            def calc_max_pain():
                max_pain_strike = mid_strike
                min_pain = float('inf')
                
                for strike in strikes:
                    pain = 0
                    for s in strikes:
                        if s in calls:
                            # Call writers profit if price < strike
                            if strike < s:
                                pain += calls[s]["oi"] * (s - strike)
                        if s in puts:
                            # Put writers profit if price > strike
                            if strike > s:
                                pain += puts[s]["oi"] * (strike - s)
                    
                    if pain < min_pain:
                        min_pain = pain
                        max_pain_strike = strike
                
                return max_pain_strike
            
            max_pain = calc_max_pain()
            signals["max_pain"] = {
                "strike": max_pain,
                "interpretation": "Likely expiry magnet zone",
                "distance_from_mid": round(max_pain - mid_strike, 0)
            }
            
            # ========== 2. PUT-CALL RATIO (PCR) ==========
            oi_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
            
            # Change in OI PCR
            total_call_chg_oi = sum(c["chg_oi"] for c in calls.values())
            total_put_chg_oi = sum(p["chg_oi"] for p in puts.values())
            chg_oi_pcr = total_put_chg_oi / total_call_chg_oi if total_call_chg_oi != 0 else 1
            
            # PCR interpretation
            if oi_pcr > 1.3:
                pcr_signal = "Bullish (High Put Writing)"
                market_sentiment = "Bullish"
            elif oi_pcr > 1.0:
                pcr_signal = "Mildly Bullish"
                market_sentiment = "Bullish"
            elif oi_pcr > 0.7:
                pcr_signal = "Neutral"
                market_sentiment = "Neutral"
            elif oi_pcr > 0.5:
                pcr_signal = "Mildly Bearish"
                market_sentiment = "Bearish"
            else:
                pcr_signal = "Bearish (High Call Writing)"
                market_sentiment = "Bearish"
            
            # Fresh positioning signal
            if chg_oi_pcr > 1.2:
                fresh_positioning = "Fresh Put Writing (Bullish)"
            elif chg_oi_pcr < 0.8:
                fresh_positioning = "Fresh Call Writing (Bearish)"
            else:
                fresh_positioning = "Balanced Positioning"
            
            signals["pcr_analysis"] = {
                "oi_pcr": round(oi_pcr, 2),
                "chg_oi_pcr": round(chg_oi_pcr, 2) if chg_oi_pcr != float('inf') else 0,
                "total_call_oi": round(total_call_oi, 0),
                "total_put_oi": round(total_put_oi, 0),
                "pcr_signal": pcr_signal,
                "market_sentiment": market_sentiment,
                "fresh_positioning": fresh_positioning
            }
            
            # ========== 3. SUPPORT & RESISTANCE LEVELS ==========
            # High PUT OI = Support, High CALL OI = Resistance
            sorted_puts = sorted(puts.items(), key=lambda x: x[1]["oi"], reverse=True)
            sorted_calls = sorted(calls.items(), key=lambda x: x[1]["oi"], reverse=True)
            
            support_levels = [(s, p["oi"]) for s, p in sorted_puts[:3]]
            resistance_levels = [(s, c["oi"]) for s, c in sorted_calls[:3]]
            
            signals["support_resistance"] = {
                "support_levels": [{"strike": s, "oi": oi} for s, oi in support_levels],
                "resistance_levels": [{"strike": s, "oi": oi} for s, oi in resistance_levels],
                "immediate_support": support_levels[0][0] if support_levels else 0,
                "immediate_resistance": resistance_levels[0][0] if resistance_levels else 0
            }
            
            # ========== 4. TREND BIAS (Price + OI) ==========
            # Compare call and put OI buildup
            call_oi_buildup = sum(c["chg_oi"] for c in calls.values() if c["chg_oi"] > 0)
            put_oi_buildup = sum(p["chg_oi"] for p in puts.values() if p["chg_oi"] > 0)
            call_oi_unwinding = abs(sum(c["chg_oi"] for c in calls.values() if c["chg_oi"] < 0))
            put_oi_unwinding = abs(sum(p["chg_oi"] for p in puts.values() if p["chg_oi"] < 0))
            
            # Trend bias logic
            if put_oi_buildup > call_oi_buildup * 1.3:
                trend_bias = "Bullish (Put Writing Dominant)"
            elif call_oi_buildup > put_oi_buildup * 1.3:
                trend_bias = "Bearish (Call Writing Dominant)"
            elif put_oi_unwinding > call_oi_unwinding * 1.3:
                trend_bias = "Bearish (Put Unwinding)"
            elif call_oi_unwinding > put_oi_unwinding * 1.3:
                trend_bias = "Bullish (Call Unwinding)"
            else:
                trend_bias = "Neutral (Balanced Activity)"
            
            signals["trend_bias"] = {
                "bias": trend_bias,
                "call_oi_buildup": round(call_oi_buildup, 0),
                "put_oi_buildup": round(put_oi_buildup, 0),
                "call_oi_unwinding": round(call_oi_unwinding, 0),
                "put_oi_unwinding": round(put_oi_unwinding, 0)
            }
            
            # ========== 5. VOLATILITY EXPECTATIONS ==========
            # ATM IV (find closest to mid strike)
            atm_strike = min(strikes, key=lambda x: abs(x - mid_strike))
            atm_call_iv = calls.get(atm_strike, {}).get("iv", 0)
            atm_put_iv = puts.get(atm_strike, {}).get("iv", 0)
            atm_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else max(atm_call_iv, atm_put_iv)
            
            # IV level interpretation
            if atm_iv > 25:
                iv_signal = "High Volatility (Fear/Uncertainty)"
                vol_expectation = "Expect large moves"
            elif atm_iv > 15:
                iv_signal = "Moderate Volatility"
                vol_expectation = "Normal market conditions"
            elif atm_iv > 10:
                iv_signal = "Low Volatility (Complacency)"
                vol_expectation = "Calm markets, potential breakout coming"
            else:
                iv_signal = "Very Low Volatility"
                vol_expectation = "Extremely calm, watch for spike"
            
            # IV Skew (OTM Put IV vs OTM Call IV)
            otm_puts = [(s, p["iv"]) for s, p in puts.items() if s < atm_strike - 200 and p["iv"] > 0]
            otm_calls = [(s, c["iv"]) for s, c in calls.items() if s > atm_strike + 200 and c["iv"] > 0]
            
            avg_otm_put_iv = np.mean([iv for _, iv in otm_puts]) if otm_puts else 0
            avg_otm_call_iv = np.mean([iv for _, iv in otm_calls]) if otm_calls else 0
            
            if avg_otm_put_iv > avg_otm_call_iv * 1.2:
                skew_signal = "Put Skew (Downside Fear)"
            elif avg_otm_call_iv > avg_otm_put_iv * 1.2:
                skew_signal = "Call Skew (Upside Speculation)"
            else:
                skew_signal = "Neutral Skew"
            
            signals["volatility"] = {
                "atm_strike": atm_strike,
                "atm_iv": round(atm_iv, 2),
                "iv_signal": iv_signal,
                "vol_expectation": vol_expectation,
                "avg_otm_put_iv": round(avg_otm_put_iv, 2),
                "avg_otm_call_iv": round(avg_otm_call_iv, 2),
                "iv_skew": skew_signal
            }
            
            # ========== 6. SMART MONEY POSITIONING ==========
            # High OI + Low IV = Writing (smart money)
            # High OI + High IV = Buying (retail)
            smart_money_calls = [(s, c) for s, c in calls.items() if c["oi"] > 100000 and c["iv"] < 15]
            smart_money_puts = [(s, p) for s, p in puts.items() if p["oi"] > 100000 and p["iv"] < 15]
            
            if len(smart_money_puts) > len(smart_money_calls):
                smart_money_signal = "Put Writing Dominant (Bullish Smart Money)"
            elif len(smart_money_calls) > len(smart_money_puts):
                smart_money_signal = "Call Writing Dominant (Bearish Smart Money)"
            else:
                smart_money_signal = "Balanced Smart Money Activity"
            
            signals["smart_money"] = {
                "signal": smart_money_signal,
                "call_writing_strikes": len(smart_money_calls),
                "put_writing_strikes": len(smart_money_puts)
            }
            
            # ========== 7. GAMMA ZONES ==========
            # High gamma = high dealer hedging = pinning/volatility
            # Gamma proxy: strikes with highest OI near ATM
            gamma_range = [s for s in strikes if abs(s - atm_strike) <= 300]
            gamma_call_oi = sum(calls.get(s, {}).get("oi", 0) for s in gamma_range)
            gamma_put_oi = sum(puts.get(s, {}).get("oi", 0) for s in gamma_range)
            total_gamma_oi = gamma_call_oi + gamma_put_oi
            
            if total_gamma_oi > 500000:
                gamma_signal = "High Gamma Zone (Pinning Likely)"
            elif total_gamma_oi > 200000:
                gamma_signal = "Moderate Gamma (Some Hedging)"
            else:
                gamma_signal = "Low Gamma (Free Price Movement)"
            
            # Potential pinning level
            max_gamma_strike = max(gamma_range, key=lambda s: calls.get(s, {}).get("oi", 0) + puts.get(s, {}).get("oi", 0)) if gamma_range else atm_strike
            
            signals["gamma_analysis"] = {
                "gamma_range": f"{min(gamma_range)}-{max(gamma_range)}" if gamma_range else "N/A",
                "total_gamma_oi": round(total_gamma_oi, 0),
                "gamma_signal": gamma_signal,
                "likely_pinning_level": max_gamma_strike
            }
            
            # ========== 8. EQUITY CORRELATION ==========
            equity_corr = {}
            try:
                # Get equity option chain signals
                eq_oc = self.extract_option_chain_signals(symbol)
                if eq_oc.get("available"):
                    eq_pcr = eq_oc.get("market_expectation", {}).get("pcr_oi", 0)
                    eq_bias = eq_oc.get("composite_options_signal", {}).get("options_bias", "")
                    
                    # Directional alignment
                    idx_bullish = "Bullish" in market_sentiment
                    eq_bullish = "Bullish" in eq_bias
                    
                    if idx_bullish and eq_bullish:
                        alignment = "High Conviction (Both Bullish)"
                        reliability = "High"
                    elif not idx_bullish and not eq_bullish:
                        alignment = "High Conviction (Both Bearish)"
                        reliability = "High"
                    elif eq_bullish and not idx_bullish:
                        alignment = "Reduced Reliability (Equity bullish, Index bearish)"
                        reliability = "Medium"
                    else:
                        alignment = "Reduced Reliability (Equity bearish, Index bullish)"
                        reliability = "Medium"
                    
                    equity_corr["alignment"] = alignment
                    equity_corr["reliability"] = reliability
                    equity_corr["index_sentiment"] = market_sentiment
                    equity_corr["equity_bias"] = eq_bias
                    
                    # Volatility filter
                    if atm_iv > 20:
                        equity_corr["vol_recommendation"] = "High Index IV - Prefer option selling/spreads"
                    elif atm_iv < 12:
                        equity_corr["vol_recommendation"] = "Low Index IV - Prefer directional buying"
                    else:
                        equity_corr["vol_recommendation"] = "Normal IV - Balanced approach"
                    
                    # Strike validation
                    eq_support = eq_oc.get("key_levels", {}).get("put_wall_strike", 0)
                    eq_resistance = eq_oc.get("key_levels", {}).get("call_wall_strike", 0)
                    
                    if support_levels and eq_support:
                        idx_support = support_levels[0][0]
                        # Check if they align (within 2% of each other proportionally)
                        equity_corr["support_alignment"] = "Aligned" if abs(eq_support - idx_support) / idx_support < 0.02 else "Different Zones"
                    
                    # Risk regime
                    if oi_pcr > 1.5 or oi_pcr < 0.5:
                        equity_corr["risk_regime"] = "Extreme PCR - Expect mean reversion, cautious entries"
                    else:
                        equity_corr["risk_regime"] = "Normal PCR - Standard risk"
                        
            except:
                pass
            
            signals["equity_correlation"] = equity_corr if equity_corr else {"available": False}
            
            # ========== 9. MULTI-EXPIRY ANALYSIS ==========
            multi_expiry = {
                "expiries_analyzed": len(expiry_data),
                "expiry_list": list(expiry_data.keys()),
                "total_call_oi_all_expiries": round(total_call_oi_all, 0),
                "total_put_oi_all_expiries": round(total_put_oi_all, 0),
                "aggregate_pcr": round(total_put_oi_all / total_call_oi_all, 2) if total_call_oi_all > 0 else 1
            }
            
            # Per-expiry breakdown
            expiry_breakdown = []
            for exp, data in expiry_data.items():
                exp_pcr = data["total_put_oi"] / data["total_call_oi"] if data["total_call_oi"] > 0 else 1
                
                # Find max OI levels for this expiry
                max_call_strike = max(data["calls"].items(), key=lambda x: x[1]["oi"])[0] if data["calls"] else 0
                max_put_strike = max(data["puts"].items(), key=lambda x: x[1]["oi"])[0] if data["puts"] else 0
                
                expiry_breakdown.append({
                    "expiry": exp,
                    "call_oi": round(data["total_call_oi"], 0),
                    "put_oi": round(data["total_put_oi"], 0),
                    "pcr": round(exp_pcr, 2),
                    "max_call_oi_strike": max_call_strike,
                    "max_put_oi_strike": max_put_strike
                })
            
            multi_expiry["expiry_breakdown"] = expiry_breakdown
            
            # Rollover analysis (compare nearest to next expiry)
            if len(expiry_breakdown) >= 2:
                near = expiry_breakdown[0]
                next_exp = expiry_breakdown[1]
                
                # OI concentration shift
                near_total = near["call_oi"] + near["put_oi"]
                next_total = next_exp["call_oi"] + next_exp["put_oi"]
                
                if next_total > near_total * 0.5:
                    rollover_signal = "Rollover Active (Significant OI in next expiry)"
                elif next_total > near_total * 0.2:
                    rollover_signal = "Early Rollover (Some OI shifting)"
                else:
                    rollover_signal = "Rollover Not Started (OI concentrated in near expiry)"
                
                # PCR trend across expiries
                if len(expiry_breakdown) >= 3:
                    pcr_list = [e["pcr"] for e in expiry_breakdown[:3]]
                    if pcr_list[0] < pcr_list[1] < pcr_list[2]:
                        term_structure = "Bullish Term Structure (PCR increasing with time)"
                    elif pcr_list[0] > pcr_list[1] > pcr_list[2]:
                        term_structure = "Bearish Term Structure (PCR decreasing with time)"
                    else:
                        term_structure = "Mixed Term Structure"
                else:
                    term_structure = "N/A (Insufficient expiries)"
                
                multi_expiry["rollover_signal"] = rollover_signal
                multi_expiry["term_structure"] = term_structure
                
                # Key levels across expiries (confluence zones)
                all_max_call_strikes = [e["max_call_oi_strike"] for e in expiry_breakdown if e["max_call_oi_strike"]]
                all_max_put_strikes = [e["max_put_oi_strike"] for e in expiry_breakdown if e["max_put_oi_strike"]]
                
                # Find confluence (strikes appearing in multiple expiries)
                from collections import Counter
                call_confluence = Counter(all_max_call_strikes).most_common(1)
                put_confluence = Counter(all_max_put_strikes).most_common(1)
                
                multi_expiry["resistance_confluence"] = call_confluence[0][0] if call_confluence else 0
                multi_expiry["support_confluence"] = put_confluence[0][0] if put_confluence else 0
            
            signals["multi_expiry_analysis"] = multi_expiry
            
            # ========== 10. COMPOSITE INDEX OPTIONS SIGNAL ==========
            bullish_count = 0
            bearish_count = 0
            
            if oi_pcr > 1.0: bullish_count += 1
            else: bearish_count += 1
            
            if "Bullish" in fresh_positioning: bullish_count += 1
            elif "Bearish" in fresh_positioning: bearish_count += 1
            
            if "Bullish" in trend_bias: bullish_count += 1
            elif "Bearish" in trend_bias: bearish_count += 1
            
            if "Put Skew" in skew_signal: bullish_count += 0.5  # Hedging, not necessarily bearish
            elif "Call Skew" in skew_signal: bullish_count += 0.5
            
            if "Put Writing" in smart_money_signal: bullish_count += 1
            elif "Call Writing" in smart_money_signal: bearish_count += 1
            
            if bullish_count > bearish_count + 1:
                options_bias = "Strong Bullish"
            elif bullish_count > bearish_count:
                options_bias = "Bullish"
            elif bearish_count > bullish_count + 1:
                options_bias = "Strong Bearish"
            elif bearish_count > bullish_count:
                options_bias = "Bearish"
            else:
                options_bias = "Neutral"
            
            signals["composite_index_options"] = {
                "bullish_factors": round(bullish_count, 1),
                "bearish_factors": round(bearish_count, 1),
                "options_bias": options_bias,
                "key_signal": f"PCR: {oi_pcr:.2f}, Max Pain: {max_pain}",
                "trade_recommendation": "Favor calls/longs" if "Bullish" in options_bias else ("Favor puts/shorts" if "Bearish" in options_bias else "Range-bound strategies")
            }
            
            return signals
            
        except Exception as e:
            signals["error"] = str(e)
            return signals
    
    # ==================== VIX HISTORY SIGNALS ====================
    
    def extract_vix_history_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from vix_history table (India VIX historical data)"""
        signals = {"available": False}
        
        try:
            cursor = self.conn.cursor()
            
            # Get VIX data ordered by date (need to parse dates)
            cursor.execute("""
                SELECT trade_date, open_value, high_value, low_value, close_value, 
                       prev_close, change_value, change_pct
                FROM vix_history
                ORDER BY id DESC
            """)
            rows = cursor.fetchall()
            
            if len(rows) < 20:
                return signals
            
            # Convert to arrays
            dates = [r[0] for r in rows]
            opens = np.array([r[1] or 0 for r in rows])
            highs = np.array([r[2] or 0 for r in rows])
            lows = np.array([r[3] or 0 for r in rows])
            closes = np.array([r[4] or 0 for r in rows])
            prev_closes = np.array([r[5] or 0 for r in rows])
            changes = np.array([r[6] or 0 for r in rows])
            change_pcts = np.array([r[7] or 0 for r in rows])
            
            signals["available"] = True
            signals["data_points"] = len(rows)
            signals["latest_date"] = dates[0]
            signals["current_vix"] = round(closes[0], 2)
            
            # ========== 1. VOLATILITY REGIME ==========
            current_vix = closes[0]
            
            if current_vix < 12:
                regime = "Low Volatility (Complacency)"
                market_mode = "Trending / Risk-On"
                risk_level = "Low"
            elif current_vix < 15:
                regime = "Normal Volatility"
                market_mode = "Balanced Market"
                risk_level = "Normal"
            elif current_vix < 20:
                regime = "Elevated Volatility"
                market_mode = "Cautious / Transitional"
                risk_level = "Moderate"
            elif current_vix < 25:
                regime = "High Volatility"
                market_mode = "Uncertainty / Risk-Off"
                risk_level = "High"
            else:
                regime = "Extreme Volatility (Fear)"
                market_mode = "Panic / Crisis Mode"
                risk_level = "Very High"
            
            signals["volatility_regime"] = {
                "current_vix": round(current_vix, 2),
                "regime": regime,
                "market_mode": market_mode,
                "risk_level": risk_level
            }
            
            # ========== 2. VIX TREND & MOMENTUM ==========
            # Calculate SMAs
            def calc_sma(arr, period):
                if len(arr) < period:
                    return np.mean(arr)
                return np.mean(arr[:period])
            
            sma_5 = calc_sma(closes, 5)
            sma_10 = calc_sma(closes, 10)
            sma_20 = calc_sma(closes, 20)
            
            # Trend direction
            if current_vix > sma_5 > sma_10:
                trend = "Rising"
                trend_signal = "Increasing Fear - Potential Downside"
            elif current_vix < sma_5 < sma_10:
                trend = "Falling"
                trend_signal = "Stability - Trend Continuation Likely"
            elif current_vix > sma_10:
                trend = "Above Average"
                trend_signal = "Elevated Uncertainty"
            else:
                trend = "Below Average"
                trend_signal = "Low Fear Environment"
            
            # Momentum (rate of change)
            if len(closes) >= 5:
                vix_roc_5d = (closes[0] - closes[4]) / closes[4] * 100 if closes[4] else 0
            else:
                vix_roc_5d = 0
            
            if vix_roc_5d > 20:
                momentum = "Strong Fear Surge"
            elif vix_roc_5d > 10:
                momentum = "Rising Fear"
            elif vix_roc_5d < -20:
                momentum = "Fear Collapse"
            elif vix_roc_5d < -10:
                momentum = "Declining Fear"
            else:
                momentum = "Stable"
            
            signals["vix_trend"] = {
                "trend": trend,
                "trend_signal": trend_signal,
                "sma_5": round(sma_5, 2),
                "sma_10": round(sma_10, 2),
                "sma_20": round(sma_20, 2),
                "roc_5d_pct": round(vix_roc_5d, 2),
                "momentum": momentum
            }
            
            # ========== 3. VIX SPIKES DETECTION ==========
            today_change = change_pcts[0]
            intraday_range = highs[0] - lows[0]
            avg_range = np.mean(highs[:20] - lows[:20])
            range_ratio = intraday_range / avg_range if avg_range else 1
            
            if abs(today_change) > 15:
                spike_signal = "Major Spike (Event Risk)"
            elif abs(today_change) > 10:
                spike_signal = "Significant Move"
            elif abs(today_change) > 5:
                spike_signal = "Moderate Move"
            else:
                spike_signal = "Normal Day"
            
            # Historical spike detection (last 20 days)
            recent_spikes = sum(1 for c in change_pcts[:20] if abs(c) > 10)
            
            signals["vix_spikes"] = {
                "today_change_pct": round(today_change, 2),
                "intraday_range": round(intraday_range, 2),
                "range_ratio": round(range_ratio, 2),
                "spike_signal": spike_signal,
                "recent_spikes_20d": recent_spikes,
                "spike_interpretation": "Elevated Event Risk" if recent_spikes > 3 else "Normal Volatility Pattern"
            }
            
            # ========== 4. MEAN REVERSION ZONES ==========
            vix_mean = np.mean(closes[:60]) if len(closes) >= 60 else np.mean(closes)
            vix_std = np.std(closes[:60]) if len(closes) >= 60 else np.std(closes)
            
            z_score = (current_vix - vix_mean) / vix_std if vix_std else 0
            
            if z_score > 2:
                mean_rev_signal = "Extreme High VIX - Potential Market Bottom"
                reversion_expected = "VIX likely to fall"
            elif z_score > 1:
                mean_rev_signal = "Elevated VIX - Above Normal"
                reversion_expected = "VIX may normalize"
            elif z_score < -2:
                mean_rev_signal = "Extreme Low VIX - Complacency Risk"
                reversion_expected = "VIX spike possible"
            elif z_score < -1:
                mean_rev_signal = "Low VIX - Below Normal"
                reversion_expected = "Watch for uptick"
            else:
                mean_rev_signal = "VIX Near Fair Value"
                reversion_expected = "No strong reversion signal"
            
            signals["mean_reversion"] = {
                "vix_mean_60d": round(vix_mean, 2),
                "vix_std_60d": round(vix_std, 2),
                "z_score": round(z_score, 2),
                "signal": mean_rev_signal,
                "reversion_expected": reversion_expected
            }
            
            # ========== 5. VOLATILITY EXPANSION / CONTRACTION ==========
            # Compare recent ATR to historical
            recent_atr = np.mean(highs[:5] - lows[:5])
            older_atr = np.mean(highs[10:20] - lows[10:20]) if len(highs) >= 20 else recent_atr
            
            atr_ratio = recent_atr / older_atr if older_atr else 1
            
            if atr_ratio > 1.5:
                vol_phase = "Expansion (Breakout Likely)"
            elif atr_ratio > 1.2:
                vol_phase = "Mild Expansion"
            elif atr_ratio < 0.7:
                vol_phase = "Contraction (Consolidation)"
            elif atr_ratio < 0.85:
                vol_phase = "Mild Contraction"
            else:
                vol_phase = "Normal Volatility"
            
            signals["vol_expansion"] = {
                "recent_atr_5d": round(recent_atr, 2),
                "older_atr_10d": round(older_atr, 2),
                "atr_ratio": round(atr_ratio, 2),
                "phase": vol_phase,
                "implication": "Price breakout incoming" if "Expansion" in vol_phase else ("Consolidation phase" if "Contraction" in vol_phase else "Normal trading")
            }
            
            # ========== 6. EQUITY CORRELATION ==========
            equity_corr = {}
            try:
                # Get equity historical signals
                hist = self.extract_historical_signals(symbol)
                if hist.get("available"):
                    stock_trend = hist.get("trend_structure", {}).get("trend", "")
                    stock_change = hist.get("trend_structure", {}).get("percent_change_20d", 0)
                    
                    # Price + VIX correlation
                    price_up = "Uptrend" in stock_trend or stock_change > 0
                    vix_down = current_vix < sma_10
                    
                    if price_up and vix_down:
                        price_vix = "Healthy Bullish (Price Up + VIX Down)"
                        quality = "Strong"
                    elif price_up and not vix_down:
                        price_vix = "Rally Under Stress (Price Up + VIX Up)"
                        quality = "Weak"
                    elif not price_up and not vix_down:
                        price_vix = "Strong Bearish (Price Down + VIX Up)"
                        quality = "Strong Downtrend"
                    else:
                        price_vix = "Exhaustion Pattern (Price Down + VIX Down)"
                        quality = "Potential Reversal"
                    
                    equity_corr["price_vix_correlation"] = price_vix
                    equity_corr["trend_quality"] = quality
                    
                    # Option strategy filter
                    if current_vix > 18:
                        equity_corr["option_strategy"] = "High VIX - Prefer option selling / hedged spreads"
                    elif current_vix < 12:
                        equity_corr["option_strategy"] = "Low VIX - Prefer directional option buying"
                    else:
                        equity_corr["option_strategy"] = "Neutral VIX - Balanced approach"
                    
                    # Stock IV vs VIX comparison
                    oc = self.extract_option_chain_signals(symbol)
                    if oc.get("available"):
                        stock_iv = oc.get("iv_analysis", {}).get("atm_iv", 0)
                        if stock_iv > 0 and current_vix > 0:
                            iv_ratio = stock_iv / current_vix
                            if iv_ratio > 1.5:
                                equity_corr["iv_comparison"] = "Stock IV >> VIX (Stock-Specific Risk)"
                            elif iv_ratio < 0.7:
                                equity_corr["iv_comparison"] = "Stock IV << VIX (Underpriced Options)"
                            else:
                                equity_corr["iv_comparison"] = "Stock IV ≈ VIX (Market-Driven Move)"
                            equity_corr["stock_iv"] = round(stock_iv, 2)
                            equity_corr["iv_ratio"] = round(iv_ratio, 2)
                    
                    # Entry timing
                    if "Uptrend" in stock_trend and trend == "Rising":
                        equity_corr["entry_timing"] = "Caution - Breakout + Rising VIX (Avoid Chasing)"
                    elif "Uptrend" in stock_trend and trend == "Falling":
                        equity_corr["entry_timing"] = "Favorable - Pullback + Falling VIX (Safer Entry)"
                    elif trend == "Rising":
                        equity_corr["entry_timing"] = "Wait - Rising VIX suggests caution"
                    else:
                        equity_corr["entry_timing"] = "Normal - VIX not signaling unusual risk"
                        
            except:
                pass
            
            signals["equity_correlation"] = equity_corr if equity_corr else {"available": False}
            
            # ========== 7. COMPOSITE VIX SIGNAL ==========
            bullish_count = 0
            bearish_count = 0
            
            if current_vix < 15: bullish_count += 1
            elif current_vix > 20: bearish_count += 1
            
            if trend == "Falling": bullish_count += 1
            elif trend == "Rising": bearish_count += 1
            
            if z_score < -1: bullish_count += 0.5  # Low VIX is bullish for stocks but risky
            elif z_score > 1: bearish_count += 1
            
            if "Contraction" in vol_phase: bullish_count += 0.5
            elif "Expansion" in vol_phase: bearish_count += 0.5
            
            if spike_signal == "Normal Day": bullish_count += 0.5
            elif "Spike" in spike_signal: bearish_count += 1
            
            if bullish_count > bearish_count + 1:
                vix_bias = "Bullish (Low Fear)"
            elif bullish_count > bearish_count:
                vix_bias = "Mildly Bullish"
            elif bearish_count > bullish_count + 1:
                vix_bias = "Bearish (High Fear)"
            elif bearish_count > bullish_count:
                vix_bias = "Mildly Bearish"
            else:
                vix_bias = "Neutral"
            
            signals["composite_vix_signal"] = {
                "bullish_factors": round(bullish_count, 1),
                "bearish_factors": round(bearish_count, 1),
                "vix_bias": vix_bias,
                "key_signal": f"VIX: {current_vix:.1f}, {regime}, {trend}",
                "market_implication": "Favor longs, low hedging cost" if "Bullish" in vix_bias else ("Favor hedges, reduce exposure" if "Bearish" in vix_bias else "Balanced positioning")
            }
            
            return signals
            
        except Exception as e:
            signals["error"] = str(e)
            return signals
    
    # ==================== BLOCK & BULK DEALS SIGNALS ====================
    
    def extract_block_bulk_deals_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from block_deals and bulk_deals tables"""
        signals = {"available": False, "block_deals": {"available": False}, "bulk_deals": {"available": False}}
        
        try:
            cursor = self.conn.cursor()
            from datetime import datetime
            from collections import Counter, defaultdict
            
            # Helper to parse dates
            def parse_date(date_str):
                try:
                    return datetime.strptime(date_str, "%d-%b-%Y")
                except:
                    try:
                        return datetime.strptime(date_str, "%d-%B-%Y")
                    except:
                        return datetime.min
            
            # ========== LOAD BLOCK DEALS ==========
            cursor.execute("""
                SELECT deal_date, client_name, deal_type, quantity, price
                FROM block_deals WHERE symbol = ?
                ORDER BY deal_date DESC
            """, (symbol,))
            block_rows = cursor.fetchall()
            
            # ========== LOAD BULK DEALS ==========
            cursor.execute("""
                SELECT deal_date, client_name, deal_type, quantity, price
                FROM bulk_deals WHERE symbol = ?
                ORDER BY deal_date DESC
            """, (symbol,))
            bulk_rows = cursor.fetchall()
            
            # Combine all deals
            all_deals = []
            for row in block_rows:
                all_deals.append({
                    "date": row[0],
                    "parsed_date": parse_date(row[0]),
                    "client": row[1],
                    "type": row[2],
                    "quantity": row[3] or 0,
                    "price": row[4] or 0,
                    "source": "block"
                })
            
            for row in bulk_rows:
                all_deals.append({
                    "date": row[0],
                    "parsed_date": parse_date(row[0]),
                    "client": row[1],
                    "type": row[2],
                    "quantity": row[3] or 0,
                    "price": row[4] or 0,
                    "source": "bulk"
                })
            
            if not all_deals:
                return signals
            
            signals["available"] = True
            signals["total_deals"] = len(all_deals)
            signals["block_count"] = len(block_rows)
            signals["bulk_count"] = len(bulk_rows)
            
            # Sort by date
            all_deals.sort(key=lambda x: x["parsed_date"], reverse=True)
            
            # ========== 1. INSTITUTIONAL ACTIVITY SIGNALS ==========
            buy_deals = [d for d in all_deals if d["type"].upper() == "BUY"]
            sell_deals = [d for d in all_deals if d["type"].upper() == "SELL"]
            
            total_buy_qty = sum(d["quantity"] or 0 for d in buy_deals)
            total_sell_qty = sum(d["quantity"] or 0 for d in sell_deals)
            total_buy_value = sum((d["quantity"] or 0) * (d["price"] or 0) for d in buy_deals)
            total_sell_value = sum((d["quantity"] or 0) * (d["price"] or 0) for d in sell_deals)
            
            net_qty = total_buy_qty - total_sell_qty
            net_value = total_buy_value - total_sell_value
            
            # Safety check: ensure net_value is not None before comparison
            if net_value is None:
                net_value = 0.0
            
            if net_value > 0:
                if net_value > 100_00_00_000:  # 100 Cr
                    activity_signal = "Strong Institutional Accumulation"
                else:
                    activity_signal = "Net Institutional Buying"
            elif net_value < 0:
                if abs(net_value) > 100_00_00_000:
                    activity_signal = "Strong Institutional Distribution"
                else:
                    activity_signal = "Net Institutional Selling"
            else:
                activity_signal = "Balanced Activity"
            
            # Repeated participation (conviction)
            client_counts = Counter(d["client"] for d in all_deals)
            repeat_clients = [(c, cnt) for c, cnt in client_counts.items() if cnt > 1]
            
            signals["institutional_activity"] = {
                "buy_deals": len(buy_deals),
                "sell_deals": len(sell_deals),
                "total_buy_qty": round(total_buy_qty, 0),
                "total_sell_qty": round(total_sell_qty, 0),
                "total_buy_value_cr": round(total_buy_value / 1_00_00_000, 2),
                "total_sell_value_cr": round(total_sell_value / 1_00_00_000, 2),
                "net_qty": round(net_qty, 0),
                "net_value_cr": round(net_value / 1_00_00_000, 2),
                "activity_signal": activity_signal,
                "repeat_participants": len(repeat_clients),
                "conviction": "High" if len(repeat_clients) > 2 else ("Moderate" if len(repeat_clients) > 0 else "Low")
            }
            
            # ========== 2. PRICE IMPACT & INTENT ==========
            # Get current market price from historical data
            current_price = None
            try:
                hist = self.extract_historical_signals(symbol)
                if hist.get("available"):
                    current_price = hist.get("trend_structure", {}).get("current_close", 0)
            except:
                pass
            
            price_intent = []
            for d in all_deals[:10]:  # Recent 10 deals
                if current_price and d["price"] > 0:
                    premium_pct = (d["price"] - current_price) / current_price * 100
                    
                    if d["type"].upper() == "BUY":
                        if premium_pct > 2:
                            intent = "Strong Bullish (Paid Premium)"
                        elif premium_pct > 0:
                            intent = "Bullish (Above Market)"
                        elif premium_pct > -2:
                            intent = "Opportunistic (Near Market)"
                        else:
                            intent = "Negotiated (Below Market)"
                    else:
                        if premium_pct < -2:
                            intent = "Distress Selling"
                        elif premium_pct < 0:
                            intent = "Controlled Exit"
                        else:
                            intent = "Profit Booking (Above Market)"
                    
                    price_intent.append({
                        "date": d["date"],
                        "type": d["type"],
                        "deal_price": d["price"],
                        "current_price": current_price,
                        "premium_pct": round(premium_pct, 2),
                        "intent": intent
                    })
            
            # Overall intent summary
            buy_premiums = [p["premium_pct"] for p in price_intent if p["type"].upper() == "BUY"]
            avg_buy_premium = np.mean(buy_premiums) if buy_premiums else 0
            
            if avg_buy_premium > 1:
                intent_signal = "Strong Bullish Intent (Buyers paying premium)"
            elif avg_buy_premium > -1:
                intent_signal = "Neutral Intent (Near market prices)"
            else:
                intent_signal = "Opportunistic Buying (Below market)"
            
            signals["price_intent"] = {
                "recent_deals": price_intent[:5],
                "avg_buy_premium_pct": round(avg_buy_premium, 2),
                "intent_signal": intent_signal,
                "current_market_price": current_price
            }
            
            # ========== 3. VOLUME & LIQUIDITY CONFIRMATION ==========
            # Get average daily volume
            avg_volume = None
            try:
                if hist.get("available"):
                    avg_volume = hist.get("volume_trend", {}).get("avg_volume", 0)
            except:
                pass
            
            if avg_volume and avg_volume > 0:
                deal_volume_analysis = []
                for d in all_deals[:10]:
                    pct_of_adv = (d["quantity"] / avg_volume) * 100
                    significance = "Very High" if pct_of_adv > 50 else ("High" if pct_of_adv > 20 else ("Moderate" if pct_of_adv > 5 else "Low"))
                    deal_volume_analysis.append({
                        "date": d["date"],
                        "quantity": d["quantity"],
                        "pct_of_adv": round(pct_of_adv, 2),
                        "significance": significance
                    })
                
                avg_pct_adv = np.mean([d["pct_of_adv"] for d in deal_volume_analysis]) if deal_volume_analysis else 0
                
                signals["volume_significance"] = {
                    "avg_daily_volume": round(avg_volume, 0),
                    "recent_deals_analysis": deal_volume_analysis[:5],
                    "avg_pct_of_adv": round(avg_pct_adv, 2),
                    "liquidity_impact": "High Impact" if avg_pct_adv > 20 else ("Moderate Impact" if avg_pct_adv > 5 else "Low Impact")
                }
            else:
                signals["volume_significance"] = {"available": False}
            
            # Clustered deals detection
            deal_dates = [d["parsed_date"] for d in all_deals]
            date_counts = Counter(d["date"] for d in all_deals)
            clustered_dates = [(date, cnt) for date, cnt in date_counts.items() if cnt >= 2]
            
            signals["clustered_deals"] = {
                "clustered_dates": len(clustered_dates),
                "max_deals_single_day": max(date_counts.values()) if date_counts else 0,
                "signal": "Strategic Position Building" if len(clustered_dates) > 1 else ("Some Clustering" if clustered_dates else "No Clustering")
            }
            
            # ========== 4. TIME-BASED SIGNALS ==========
            # Check if deals occurred near highs/lows
            time_signals = {}
            if current_price:
                try:
                    # Get 52-week high/low from historical stats
                    stats = self.extract_historical_stats_signals(symbol)
                    if stats.get("available"):
                        high_52w = stats.get("price_range", {}).get("high_52w", 0)
                        low_52w = stats.get("price_range", {}).get("low_52w", 0)
                        
                        recent_buys_near_low = 0
                        recent_sells_near_high = 0
                        
                        for d in buy_deals:
                            if low_52w > 0 and d["price"] < low_52w * 1.1:  # Within 10% of low
                                recent_buys_near_low += 1
                        
                        for d in sell_deals:
                            if high_52w > 0 and d["price"] > high_52w * 0.9:  # Within 10% of high
                                recent_sells_near_high += 1
                        
                        if recent_buys_near_low > 0:
                            time_signals["value_accumulation"] = f"{recent_buys_near_low} buy deals near 52w low (Value Accumulation)"
                        if recent_sells_near_high > 0:
                            time_signals["profit_booking"] = f"{recent_sells_near_high} sell deals near 52w high (Profit Booking)"
                except:
                    pass
            
            signals["time_based_signals"] = time_signals if time_signals else {"note": "No significant time-based patterns"}
            
            # ========== 5. CORRELATION WITH EXISTING DATA ==========
            correlation = {}
            try:
                # Safety check: ensure net_value is not None
                if net_value is None:
                    net_value = 0.0
                
                # Price-Volume correlation
                pv = self.extract_price_volume_signals(symbol)
                if pv.get("available") and net_value > 0:
                    delivery = pv.get("delivery_analysis", {}).get("delivery_trend", "")
                    if "High" in delivery:
                        correlation["price_volume"] = "Block Buy + High Delivery = Strong Accumulation"
                    else:
                        correlation["price_volume"] = "Block Buy but Weak Delivery"
                
                # Futures OI correlation
                fut = self.extract_futures_data_signals(symbol)
                if fut.get("available") and net_value > 0:
                    oi_signal = fut.get("positioning_sentiment", {}).get("oi_signal", "")
                    if "Long Buildup" in oi_signal:
                        correlation["futures_oi"] = "Block Buy + Long Buildup = Strong Bullish"
                    elif "Short" in oi_signal:
                        correlation["futures_oi"] = "Block Buy but Shorts Building (Caution)"
                
                # Options correlation
                oc = self.extract_option_chain_signals(symbol)
                if oc.get("available") and net_value > 0:
                    pcr = oc.get("market_expectation", {}).get("pcr_oi", 0)
                    if pcr > 1:
                        correlation["options"] = "Block Buy + High PCR = Bullish Confirmation"
                    else:
                        correlation["options"] = "Block Buy but Low PCR (Mixed Signal)"
                
                # FII/DII flow correlation
                fii = self.extract_fii_dii_signals(symbol)
                if fii.get("available"):
                    inst_bias = fii.get("composite_fii_dii_signal", {}).get("institutional_bias", "")
                    if "Bullish" in inst_bias and net_value > 0:
                        correlation["fii_dii"] = "Block Buy + FII/DII Bullish = Aligned Flows"
                    elif "Bearish" in inst_bias and net_value > 0:
                        correlation["fii_dii"] = "Block Buy but FII/DII Bearish (Divergence)"
                
                # VIX correlation
                vix = self.extract_vix_history_signals(symbol)
                if vix.get("available") and net_value > 0:
                    vix_level = vix.get("volatility_regime", {}).get("current_vix", 0)
                    if vix_level > 20:
                        correlation["vix"] = f"Block Buy at High VIX ({vix_level:.1f}) = Contrarian Bullish"
                    else:
                        correlation["vix"] = f"Block Buy at Normal VIX ({vix_level:.1f}) = Trend Following"
                        
            except:
                pass
            
            signals["data_correlation"] = correlation if correlation else {"note": "No significant correlations detected"}
            
            # ========== 6. COMPOSITE BLOCK/BULK SIGNAL ==========
            bullish_count = 0
            bearish_count = 0
            
            # Safety check: ensure net_value is not None before comparisons
            if net_value is None:
                net_value = 0.0
            
            if net_value > 50_00_00_000: bullish_count += 1  # >50 Cr net buy
            elif net_value < -50_00_00_000: bearish_count += 1
            
            if net_value > 0: bullish_count += 0.5
            elif net_value < 0: bearish_count += 0.5
            
            if avg_buy_premium > 0: bullish_count += 1
            elif avg_buy_premium < -2: bearish_count += 0.5
            
            # Safety check for net_value comparison
            if len(repeat_clients) > 1: 
                if net_value > 0:
                    bullish_count += 0.5
                else:
                    bearish_count += 0.5
            
            if "Accumulation" in activity_signal: bullish_count += 1
            elif "Distribution" in activity_signal: bearish_count += 1
            
            if bullish_count > bearish_count + 1:
                deals_bias = "Strong Bullish"
            elif bullish_count > bearish_count:
                deals_bias = "Bullish"
            elif bearish_count > bullish_count + 1:
                deals_bias = "Strong Bearish"
            elif bearish_count > bullish_count:
                deals_bias = "Bearish"
            else:
                deals_bias = "Neutral"
            
            signals["composite_deals_signal"] = {
                "bullish_factors": round(bullish_count, 1),
                "bearish_factors": round(bearish_count, 1),
                "deals_bias": deals_bias,
                "key_signal": activity_signal,
                "recommendation": "Institutional support present" if "Bullish" in deals_bias else ("Watch for distribution" if "Bearish" in deals_bias else "Monitor for direction")
            }
            
            # Store block/bulk specific summaries
            signals["block_deals"]["available"] = len(block_rows) > 0
            signals["block_deals"]["count"] = len(block_rows)
            signals["bulk_deals"]["available"] = len(bulk_rows) > 0
            signals["bulk_deals"]["count"] = len(bulk_rows)
            
            return signals
            
        except Exception as e:
            signals["error"] = str(e)
            return signals
    
    # ==================== RESULTS COMPARISON SIGNALS ====================
    
    def extract_results_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from results_comparison data"""
        results = self._get_layer1_json(symbol, "results_comparison")
        signals = {"available": False}
        
        if not results:
            return signals
        
        # Handle dict wrapper (NSE format: {resCmpData: [...], bankNonBnking: ...})
        if isinstance(results, dict):
            # Safely get the results list with multiple fallbacks
            res_cmp_data = results.get("resCmpData") if isinstance(results.get("resCmpData"), list) else None
            data_field = results.get("data") if isinstance(results.get("data"), list) else None
            results = res_cmp_data or data_field or []
        
        if not isinstance(results, list) or len(results) < 2:
            return signals
        
        signals["available"] = True
        
        # Get recent quarters (assuming sorted newest first)
        quarters = results[:8] if len(results) >= 8 else results
        
        # Extract key metrics across quarters
        def safe_float(val):
            try:
                if val is None or val == '' or val == '-':
                    return 0.0
                return float(str(val).replace(',', ''))
            except (ValueError, TypeError):
                return 0.0
        
        # Safely extract metrics with multiple fallback keys
        net_sales = []
        net_profit = []
        eps = []
        
        for q in quarters:
            if not isinstance(q, dict):
                net_sales.append(0.0)
                net_profit.append(0.0)
                eps.append(0.0)
                continue
            
            # Net sales with multiple key fallbacks
            ns = safe_float(q.get('re_net_sale'))
            if ns == 0:
                ns = safe_float(q.get('re_net_sales'))
            if ns == 0:
                ns = safe_float(q.get('net_sales'))
            net_sales.append(ns)
            
            # Net profit with multiple key fallbacks
            np_val = safe_float(q.get('re_con_pro_loss'))
            if np_val == 0:
                np_val = safe_float(q.get('re_pro_loss_ord_act_tax'))
            if np_val == 0:
                np_val = safe_float(q.get('net_profit'))
            net_profit.append(np_val)
            
            # EPS with multiple key fallbacks
            eps_val = safe_float(q.get('re_diluted_eps'))
            if eps_val == 0:
                eps_val = safe_float(q.get('re_basic_eps_for_cont_dic_opr'))
            if eps_val == 0:
                eps_val = safe_float(q.get('eps'))
            eps.append(eps_val)
        total_expenses = []
        operating_income = []
        tax = []
        pbt = []
        
        for i, q in enumerate(quarters):
            if not isinstance(q, dict):
                total_expenses.append(0.0)
                operating_income.append(0.0)
                tax.append(0.0)
                pbt.append(0.0)
                continue
            
            # Total expenses
            te = safe_float(q.get('re_tot_exp_exc_pro_cont'))
            if te == 0:
                te = safe_float(q.get('re_total_expenses'))
            total_expenses.append(te)
            
            # Operating income
            oi = safe_float(q.get('re_income_inv', 0))
            ns_val = safe_float(q.get('re_net_sale', 0))
            operating_income.append(oi + ns_val)
            
            # Tax
            tax.append(safe_float(q.get('re_deff_tax', 0)))
            
            # PBT (Profit Before Tax)
            pbt_val = safe_float(q.get('re_pro_loss_bfr_tax'))
            if pbt_val == 0:
                pbt_val = net_profit[i] + tax[i]
            pbt.append(pbt_val)
        
        operating_profit = [operating_income[i] - total_expenses[i] for i in range(len(quarters))]
        
        # 1. Earnings & Profitability
        # QoQ growth
        qoq_revenue_growth = ((net_sales[0] - net_sales[1]) / net_sales[1] * 100) if len(net_sales) >= 2 and net_sales[1] > 0 else 0
        qoq_profit_growth = ((net_profit[0] - net_profit[1]) / abs(net_profit[1]) * 100) if len(net_profit) >= 2 and net_profit[1] != 0 else 0
        
        # YoY growth (Q vs Q-4)
        yoy_revenue_growth = ((net_sales[0] - net_sales[4]) / net_sales[4] * 100) if len(net_sales) >= 5 and net_sales[4] > 0 else 0
        yoy_profit_growth = ((net_profit[0] - net_profit[4]) / abs(net_profit[4]) * 100) if len(net_profit) >= 5 and net_profit[4] != 0 else 0
        
        # EPS trend
        eps_growth = ((eps[0] - eps[1]) / abs(eps[1]) * 100) if len(eps) >= 2 and eps[1] != 0 else 0
        
        # Margins
        net_margin = (net_profit[0] / net_sales[0] * 100) if net_sales[0] > 0 else 0
        operating_margin = (operating_profit[0] / net_sales[0] * 100) if net_sales[0] > 0 else 0
        
        # Tax efficiency
        tax_rate = (tax[0] / pbt[0] * 100) if pbt[0] > 0 else 0
        
        # EPS Trend calculation (based on last 4 quarters)
        if len(eps) >= 4:
            eps_increasing = sum(1 for i in range(len(eps)-1) if eps[i] > eps[i+1])
            eps_trend = "Improving" if eps_increasing >= 3 else ("Declining" if eps_increasing <= 1 else "Mixed")
        else:
            eps_trend = "Improving" if eps_growth > 5 else ("Declining" if eps_growth < -5 else "Stable")
        
        # Profitability Rating (based on margin quality and growth)
        profitability_score = 0
        if net_margin > 15: profitability_score += 2
        elif net_margin > 10: profitability_score += 1
        elif net_margin < 5: profitability_score -= 1
        
        if operating_margin > 20: profitability_score += 2
        elif operating_margin > 12: profitability_score += 1
        elif operating_margin < 8: profitability_score -= 1
        
        if qoq_profit_growth > 10: profitability_score += 1
        elif qoq_profit_growth < -10: profitability_score -= 1
        
        if yoy_profit_growth > 15: profitability_score += 1
        elif yoy_profit_growth < -15: profitability_score -= 1
        
        if profitability_score >= 4:
            profitability_rating = "Excellent"
        elif profitability_score >= 2:
            profitability_rating = "Good"
        elif profitability_score >= 0:
            profitability_rating = "Average"
        elif profitability_score >= -2:
            profitability_rating = "Below Average"
        else:
            profitability_rating = "Poor"
        
        signals["earnings_profitability"] = {
            "latest_net_sales": net_sales[0],
            "latest_net_profit": net_profit[0],
            "latest_eps": eps[0],
            "qoq_revenue_growth_pct": round(qoq_revenue_growth, 2),
            "qoq_profit_growth_pct": round(qoq_profit_growth, 2),
            "yoy_revenue_growth_pct": round(yoy_revenue_growth, 2),
            "yoy_profit_growth_pct": round(yoy_profit_growth, 2),
            "eps_growth_qoq_pct": round(eps_growth, 2),
            "eps_trend": eps_trend,
            "net_profit_margin_pct": round(net_margin, 2),
            "operating_margin_pct": round(operating_margin, 2),
            "effective_tax_rate_pct": round(tax_rate, 2),
            "profitability_rating": profitability_rating,
            "earnings_trend": "Improving" if qoq_profit_growth > 0 and yoy_profit_growth > 0 else ("Declining" if qoq_profit_growth < 0 and yoy_profit_growth < 0 else "Mixed")
        }
        
        # 2. Revenue Quality
        expense_ratio = (total_expenses[0] / net_sales[0] * 100) if net_sales[0] > 0 else 0
        
        # Expense trend
        expense_trend = "Increasing" if len(total_expenses) >= 2 and total_expenses[0] > total_expenses[1] else "Decreasing"
        
        signals["revenue_quality"] = {
            "expense_to_sales_ratio_pct": round(expense_ratio, 2),
            "expense_trend": expense_trend,
            "sales_consistency": "Stable" if all(s > 0 for s in net_sales[:4]) else "Volatile"
        }
        
        # 3. Financial Health
        # Interest coverage (if available)
        interest = safe_float(quarters[0].get('re_int_cst', 0))
        interest_coverage = (pbt[0] + interest) / interest if interest > 0 else float('inf')
        
        # Depreciation trend
        depreciation = [safe_float(q.get('re_deprn_amort', 0)) for q in quarters]
        
        signals["financial_health"] = {
            "interest_expense": interest,
            "interest_coverage_ratio": round(interest_coverage, 2) if interest_coverage != float('inf') else "No Debt",
            "depreciation_latest": depreciation[0] if depreciation else 0,
            "debt_burden": "Low" if interest_coverage > 5 or interest_coverage == float('inf') else ("High" if interest_coverage < 2 else "Moderate")
        }
        
        # 4. Growth & Stability Signals
        # Earnings consistency (variance)
        profit_variance = np.std(net_profit[:4]) / np.mean(net_profit[:4]) * 100 if len(net_profit) >= 4 and np.mean(net_profit[:4]) != 0 else 0
        
        # Quarterly acceleration
        if len(net_profit) >= 3:
            recent_growth = net_profit[0] - net_profit[1]
            prior_growth = net_profit[1] - net_profit[2]
            acceleration = "Accelerating" if recent_growth > prior_growth else ("Decelerating" if recent_growth < prior_growth else "Stable")
        else:
            acceleration = "N/A"
        
        signals["growth_stability"] = {
            "earnings_variance_pct": round(profit_variance, 2),
            "earnings_consistency": "High" if profit_variance < 20 else ("Low" if profit_variance > 50 else "Medium"),
            "quarterly_acceleration": acceleration,
            "growth_quality": "Clean" if all(safe_float(q.get('re_excpt_items', 0)) == 0 for q in quarters[:4]) else "Contains Exceptionals"
        }
        
        return signals
    
    # ==================== HISTORICAL STATS (INDIANAPI) SIGNALS ====================
    
    def extract_historical_stats_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from historical_stats_indianapi data"""
        stats = self._get_layer1_json(symbol, "historical_stats_indianapi")
        signals = {"available": False}
        
        if not stats or not isinstance(stats, dict):
            return signals
        
        def safe_float(val):
            try:
                if val is None or val == '' or val == '-':
                    return 0.0
                return float(str(val).replace(',', '').replace('%', ''))
            except (ValueError, TypeError):
                return 0.0
        
        # Handle IndianAPI structure: {metric_name: {quarter: value, ...}, ...}
        # Extract values in chronological order (newest first)
        def extract_metric(metric_name):
            metric_data = stats.get(metric_name, {})
            if isinstance(metric_data, dict):
                # Sort quarters by date (newest first)
                quarters = sorted(metric_data.keys(), reverse=True)
                return [safe_float(metric_data.get(q, 0)) for q in quarters]
            return []
        
        sales = extract_metric('Sales')
        expenses = extract_metric('Expenses')
        operating_profit = extract_metric('Operating Profit')
        opm = extract_metric('OPM %')
        other_income = extract_metric('Other Income')
        interest = extract_metric('Interest')
        depreciation = extract_metric('Depreciation')
        pbt = extract_metric('Profit before tax')
        tax_pct = extract_metric('Tax %')
        net_profit = extract_metric('Net Profit')
        
        if not sales or len(sales) < 2:
            return signals
        
        signals["available"] = True
        
        # 1. Growth & Trend
        # Revenue CAGR (if enough data)
        # Filter out zero/negative values for CAGR calculation
        valid_sales = [s for s in sales if s and s > 0]
        if len(valid_sales) >= 4 and valid_sales[-1] > 0 and valid_sales[0] > 0:
            cagr_periods = len(valid_sales) - 1
            revenue_cagr = ((valid_sales[0] / valid_sales[-1]) ** (4 / cagr_periods) - 1) * 100
        else:
            revenue_cagr = 0
        
        # QoQ/YoY growth
        qoq_sales = ((sales[0] - sales[1]) / sales[1] * 100) if len(sales) >= 2 and sales[1] > 0 else 0
        yoy_sales = ((sales[0] - sales[4]) / sales[4] * 100) if len(sales) >= 5 and sales[4] > 0 else 0
        
        qoq_profit = ((net_profit[0] - net_profit[1]) / abs(net_profit[1]) * 100) if len(net_profit) >= 2 and net_profit[1] != 0 else 0
        yoy_profit = ((net_profit[0] - net_profit[4]) / abs(net_profit[4]) * 100) if len(net_profit) >= 5 and net_profit[4] != 0 else 0
        
        # Profit trend
        profit_trend = "Up" if len(net_profit) >= 2 and net_profit[0] > net_profit[1] else "Down"
        
        signals["growth_trend"] = {
            "revenue_cagr_pct": round(revenue_cagr, 2),
            "sales_qoq_pct": round(qoq_sales, 2),
            "sales_yoy_pct": round(yoy_sales, 2),
            "profit_qoq_pct": round(qoq_profit, 2),
            "profit_yoy_pct": round(yoy_profit, 2),
            "profit_trend": profit_trend,
            "growth_phase": "High Growth" if yoy_sales > 15 else ("Moderate" if yoy_sales > 5 else ("Declining" if yoy_sales < 0 else "Stable"))
        }
        
        # 2. Profitability
        avg_opm = np.mean([o for o in opm if o > 0]) if any(o > 0 for o in opm) else 0
        opm_trend = "Improving" if len(opm) >= 2 and opm[0] > opm[1] else "Declining"
        
        net_margin = (net_profit[0] / sales[0] * 100) if len(sales) > 0 and sales[0] > 0 else 0
        
        signals["profitability"] = {
            "latest_opm_pct": opm[0] if opm else 0,
            "avg_opm_pct": round(avg_opm, 2),
            "opm_trend": opm_trend,
            "net_profit_margin_pct": round(net_margin, 2),
            "profitability_grade": "A" if net_margin > 15 else ("B" if net_margin > 10 else ("C" if net_margin > 5 else "D"))
        }
        
        # 3. Quality of Earnings
        other_inc_val = other_income[0] if other_income else 0
        core_income_ratio = ((sales[0] - other_inc_val) / sales[0] * 100) if len(sales) > 0 and sales[0] > 0 else 100
        
        signals["earnings_quality"] = {
            "core_income_ratio_pct": round(core_income_ratio, 2),
            "other_income_dependency": "Low" if core_income_ratio > 95 else ("High" if core_income_ratio < 80 else "Medium"),
            "earnings_consistency": "Stable" if (len(net_profit) >= 4 and np.mean(net_profit[:4]) != 0 and np.std(net_profit[:4]) / abs(np.mean(net_profit[:4])) < 0.3) else "Volatile"
        }
        
        # 4. Capital & Risk Signals
        depr_val = depreciation[0] if depreciation else 0
        capex_intensity = (depr_val / sales[0] * 100) if len(sales) > 0 and sales[0] > 0 else 0
        tax_stability = np.std(tax_pct[:4]) / abs(np.mean(tax_pct[:4])) if len(tax_pct) >= 4 and np.mean(tax_pct[:4]) != 0 else 0
        
        signals["capital_risk"] = {
            "capex_intensity_pct": round(capex_intensity, 2),
            "depreciation_trend": "Increasing" if len(depreciation) >= 2 and depreciation[0] > depreciation[1] else "Stable",
            "tax_stability": "Stable" if tax_stability < 0.3 else "Volatile",
            "investment_phase": "Expansion" if capex_intensity > 5 else "Mature"
        }
        
        return signals
    
    # ==================== STOCK DETAILS (INDIANAPI) SIGNALS ====================
    
    def extract_stock_details_signals(self, symbol: str) -> Dict[str, Any]:
        """Extract signals from stock_details_indianapi (analyst views, risk, news, shareholding)"""
        details = self._get_layer1_json(symbol, "stock_details_indianapi")
        signals = {"available": False}
        
        if not details or not isinstance(details, dict):
            return signals
        
        signals["available"] = True
        
        def safe_float(val):
            try:
                if val is None or val == '' or val == '-':
                    return 0.0
                return float(str(val).replace(',', ''))
            except (ValueError, TypeError):
                return 0.0
        
        def extract_metric(metrics_list, key_contains):
            """Extract value from keyMetrics list by key substring"""
            if not metrics_list or not isinstance(metrics_list, list):
                return None
            for item in metrics_list:
                if not isinstance(item, dict):
                    continue
                item_key = item.get("key", "")
                if item_key and key_contains.lower() in item_key.lower():
                    return safe_float(item.get("value"))
            return None
        
        # 1. Analyst Recommendations
        analyst_view = details.get("analystView", [])
        if analyst_view and isinstance(analyst_view, list):
            ratings = {}
            for av in analyst_view:
                if not isinstance(av, dict):
                    continue
                name = av.get("ratingName", "")
                if name and name != "Total":
                    ratings[name.lower().replace(" ", "_")] = int(safe_float(av.get("numberOfAnalystsLatest", 0)))
            
            total = sum(ratings.values())
            buy_count = ratings.get("strong_buy", 0) + ratings.get("buy", 0)
            sell_count = ratings.get("strong_sell", 0) + ratings.get("sell", 0)
            
            if total > 0:
                buy_pct = (buy_count / total) * 100
                sell_pct = (sell_count / total) * 100
            else:
                buy_pct = sell_pct = 0
            
            signals["analyst_recommendations"] = {
                "strong_buy": ratings.get("strong_buy", 0),
                "buy": ratings.get("buy", 0),
                "hold": ratings.get("hold", 0),
                "sell": ratings.get("sell", 0),
                "strong_sell": ratings.get("strong_sell", 0),
                "total_analysts": total,
                "buy_pct": round(buy_pct, 1),
                "sell_pct": round(sell_pct, 1),
                "consensus": "Strong Buy" if buy_pct > 80 else ("Buy" if buy_pct > 60 else ("Hold" if buy_pct > 40 else "Sell"))
            }
        
        # Recommendation bar summary
        recos_bar = details.get("recosBar", {})
        if recos_bar and isinstance(recos_bar, dict):
            # Ensure analyst_recommendations exists
            if "analyst_recommendations" not in signals:
                signals["analyst_recommendations"] = {"strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0, "total_analysts": 0, "buy_pct": 0, "sell_pct": 0, "consensus": "Unknown"}
            signals["analyst_recommendations"]["mean_rating"] = round(safe_float(recos_bar.get("meanValue")), 2)
            signals["analyst_recommendations"]["bullish_pct"] = round(safe_float(recos_bar.get("tickerPercentage")), 1)
        
        # 2. Risk Assessment
        risk_meter = details.get("riskMeter", {})
        if risk_meter and isinstance(risk_meter, dict):
            signals["risk_assessment"] = {
                "risk_category": risk_meter.get("categoryName", "Unknown"),
                "std_deviation": safe_float(risk_meter.get("stdDev")),
                "risk_level": "High" if safe_float(risk_meter.get("stdDev")) > 25 else ("Low" if safe_float(risk_meter.get("stdDev")) < 15 else "Moderate")
            }
        
        # 3. Key Metrics extraction
        key_metrics = details.get("keyMetrics", {})
        if not isinstance(key_metrics, dict):
            key_metrics = {}
        
        # Management Effectiveness
        mgmt = key_metrics.get("mgmtEffectiveness", [])
        signals["management_effectiveness"] = {
            "roe_5yr_avg": extract_metric(mgmt, "returnOnAverageEquity5Year"),
            "roe_ttm": extract_metric(mgmt, "returnOnAverageEquityTrailing"),
            "roi_ttm": extract_metric(mgmt, "returnOnInvestmentTrailing"),
            "roa_ttm": extract_metric(mgmt, "returnOnAverageAssetsTrailing"),
            "asset_turnover": extract_metric(mgmt, "assetTurnoverTrailing"),
            "inventory_turnover": extract_metric(mgmt, "inventoryTurnoverTrailing")
        }
        
        # Margins
        margins = key_metrics.get("margins", [])
        signals["margins"] = {
            "gross_margin_ttm": extract_metric(margins, "grossMarginTrailing"),
            "gross_margin_5yr": extract_metric(margins, "grossMargin5Year"),
            "operating_margin_ttm": extract_metric(margins, "operatingMarginTrailing"),
            "operating_margin_5yr": extract_metric(margins, "operatingMargin5Year"),
            "net_profit_margin_ttm": extract_metric(margins, "netProfitMarginPercentTrailing"),
            "net_profit_margin_5yr": extract_metric(margins, "netProfitMargin5Year"),
            "pretax_margin_ttm": extract_metric(margins, "pretaxMarginTrailing")
        }
        
        # Financial Strength
        fin_strength = key_metrics.get("financialstrength", [])
        signals["financial_strength"] = {
            "current_ratio": extract_metric(fin_strength, "currentRatioMostRecentQuarter"),
            "quick_ratio": extract_metric(fin_strength, "quickRatioMostRecentQuarter"),
            "debt_to_equity": extract_metric(fin_strength, "totalDebtPerTotalEquityMostRecentQuarter"),
            "lt_debt_to_equity": extract_metric(fin_strength, "lTDebtPerEquityMostRecentQuarter"),
            "interest_coverage": extract_metric(fin_strength, "netInterestCoverage"),
            "payout_ratio": extract_metric(fin_strength, "payoutRatioTrailing"),
            "free_cash_flow": extract_metric(fin_strength, "freeCashFlowMostRecent")
        }
        
        # Valuation
        valuation = key_metrics.get("valuation", [])
        if not isinstance(valuation, list):
            valuation = []
        peg_val = extract_metric(valuation, "pegRatio")
        signals["valuation_metrics"] = {
            "pe_ttm": extract_metric(valuation, "pPerEBasicExcludingExtraordinaryItemsTTM"),
            "pe_fy": extract_metric(valuation, "pPerEExcludingExtraordinaryItemsMostRecentFiscalYear"),
            "peg_ratio": peg_val,
            "price_to_book": extract_metric(valuation, "priceToBookMostRecentQuarter"),
            "price_to_sales": extract_metric(valuation, "priceToSalesTrailing"),
            "price_to_cash_flow": extract_metric(valuation, "priceToCashFlowPerShare"),
            "dividend_yield": extract_metric(valuation, "currentDividendYield"),
            "dividend_yield_5yr": extract_metric(valuation, "dividendYield5Year"),
            "ev_commentary": "Expensive" if peg_val and peg_val > 2 else "Fairly Valued" if peg_val and peg_val > 1 else "Undervalued"
        }
        
        # Growth Metrics
        growth = key_metrics.get("growth", [])
        signals["growth_metrics"] = {
            "revenue_growth_5yr": extract_metric(growth, "revenueGrowthRate5Year"),
            "revenue_growth_3yr": extract_metric(growth, "growthRatePercentRevenue3Year"),
            "eps_growth_5yr": extract_metric(growth, "ePSGrowthRate5Year"),
            "eps_growth_3yr": extract_metric(growth, "growthRatePercentEPS3year"),
            "eps_change_ttm": extract_metric(growth, "ePSChangePercentTTM"),
            "dividend_growth_3yr": extract_metric(growth, "growthRatePercentDividend3Year"),
            "ebitda_cagr_5yr": extract_metric(growth, "earningsBeforeInterestTaxesDepreciationAmortization5YearCAGR"),
            "book_value_growth_5yr": extract_metric(growth, "bookValuePerShareGrowthRate5Year"),
            "capex_growth_5yr": extract_metric(growth, "capitalSpendingGrowthRate5Year")
        }
        
        # Per Share Data
        pershare = key_metrics.get("persharedata", [])
        # Try multiple key patterns for EPS (data has typos like "eEPSExcludingExtraordinaryIitemsTrailing12onth")
        eps_ttm = extract_metric(pershare, "epsexcludingextra") or extract_metric(pershare, "epsincludingextra") or extract_metric(pershare, "epstrailing")
        signals["per_share_data"] = {
            "eps_ttm": eps_ttm,
            "book_value": extract_metric(pershare, "bookValuePerShareMostRecentQuarter"),
            "cash_per_share": extract_metric(pershare, "cashPerShareMostRecentQuarter"),
            "revenue_per_share": extract_metric(pershare, "revenuePerShare"),
            "dividend_per_share": extract_metric(pershare, "dividendsPerShare"),
            "cash_flow_per_share": extract_metric(pershare, "cashFlowPerShare")
        }
        
        # Price and Volume metrics
        price_vol = key_metrics.get("priceandVolume", [])
        signals["price_volume_metrics"] = {
            "market_cap": extract_metric(price_vol, "marketCap"),
            "beta": extract_metric(price_vol, "beta"),
            "price_change_1d": extract_metric(price_vol, "price1DayPercentChange"),
            "price_change_5d": extract_metric(price_vol, "price5DayPercentChange"),
            "price_change_13w": extract_metric(price_vol, "price13WeekPricePercentChange"),
            "price_change_26w": extract_metric(price_vol, "price26WeekPricePercentChange"),
            "price_change_52w": extract_metric(price_vol, "price52WeekPricePercentChange"),
            "price_change_ytd": extract_metric(price_vol, "priceYTDPricePercentChange"),
            "relative_strength_13w": extract_metric(price_vol, "relativePricePercentChange13Week"),
            "relative_strength_52w": extract_metric(price_vol, "relativePricePercentChange52Week")
        }
        
        # 4. Technical Moving Averages
        tech_data = details.get("stockTechnicalData", [])
        if tech_data and isinstance(tech_data, list):
            signals["moving_averages"] = {}
            for t in tech_data:
                if not isinstance(t, dict):
                    continue
                days = t.get("days")
                price = safe_float(t.get("nsePrice", t.get("bsePrice")))
                if days and price:
                    signals["moving_averages"][f"sma_{days}"] = price
        
        # 5. Shareholding Pattern
        shareholding = details.get("shareholding", [])
        if shareholding and isinstance(shareholding, list):
            signals["shareholding"] = {}
            for category in shareholding:
                if not isinstance(category, dict):
                    continue
                cat_name = category.get("displayName", "").lower()
                holdings = category.get("categories", [])
                if holdings and isinstance(holdings, list):
                    latest = holdings[-1] if holdings else {}
                    prev = holdings[-2] if len(holdings) > 1 else {}
                    
                    latest_pct = safe_float(latest.get("percentage"))
                    prev_pct = safe_float(prev.get("percentage"))
                    change = latest_pct - prev_pct
                    
                    signals["shareholding"][cat_name] = {
                        "current_pct": latest_pct,
                        "previous_pct": prev_pct,
                        "change": round(change, 2),
                        "trend": "Increasing" if change > 0.1 else ("Decreasing" if change < -0.1 else "Stable"),
                        "as_of": latest.get("holdingDate")
                    }
            
            # FII/DII sentiment
            fii = signals["shareholding"].get("fii", {})
            mf = signals["shareholding"].get("mf", {})
            if fii and mf:
                fii_trend = fii.get("change", 0)
                mf_trend = mf.get("change", 0)
                if fii_trend > 0 and mf_trend > 0:
                    inst_sentiment = "Strong Institutional Buying"
                elif fii_trend < 0 and mf_trend < 0:
                    inst_sentiment = "Institutional Selling"
                elif mf_trend > 0:
                    inst_sentiment = "Domestic Institutional Buying"
                elif fii_trend > 0:
                    inst_sentiment = "Foreign Institutional Buying"
                else:
                    inst_sentiment = "Mixed Institutional Activity"
                signals["shareholding"]["institutional_sentiment"] = inst_sentiment
        
        # 6. Recent News (for sentiment)
        news = details.get("recentNews", [])
        if news:
            signals["recent_news"] = []
            for n in news[:7]:  # Last 7 news items
                headline = n.get("headline", n.get("title", ""))
                date = n.get("date", "")
                
                # Simple sentiment detection
                headline_lower = headline.lower()
                if any(w in headline_lower for w in ["gain", "surge", "jump", "rise", "profit", "growth", "buy", "upgrade", "positive", "record"]):
                    sentiment = "Positive"
                elif any(w in headline_lower for w in ["fall", "drop", "decline", "loss", "sell", "downgrade", "negative", "claim", "dispute", "problem"]):
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                
                signals["recent_news"].append({
                    "headline": headline[:150],  # Truncate long headlines
                    "date": date,
                    "sentiment": sentiment
                })
            
            # News sentiment summary
            sentiments = [n["sentiment"] for n in signals["recent_news"]]
            pos_count = sentiments.count("Positive")
            neg_count = sentiments.count("Negative")
            signals["news_sentiment"] = {
                "positive_count": pos_count,
                "negative_count": neg_count,
                "neutral_count": sentiments.count("Neutral"),
                "overall": "Bullish" if pos_count > neg_count + 1 else ("Bearish" if neg_count > pos_count + 1 else "Mixed")
            }
        
        # 7. Company Profile
        signals["company_profile"] = {
            "name": details.get("companyName"),
            "industry": details.get("industry"),
            "profile": str(details.get("companyProfile", ""))[:500] if details.get("companyProfile") else None
        }
        
        return signals
    
    # ==================== COMPILE ALL SIGNALS ====================
    
    def compile_signals(self, symbol: str) -> Dict[str, Any]:
        """Compile all signals into a single JSON for LLM consumption"""
        symbol = symbol.upper()
        
        compiled = {
            "meta": {
                "symbol": symbol,
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        # Extract all signal categories
        print(f"Extracting signals for {symbol}...")
        
        print("  - Introduction...")
        compiled["introduction"] = self.extract_introduction(symbol)
        
        print("  - Intraday signals...")
        compiled["intraday_signals"] = self.extract_intraday_signals(symbol)
        
        print("  - Historical signals...")
        compiled["historical_signals"] = self.extract_historical_signals(symbol)
        
        print("  - Advanced price-volume signals...")
        compiled["price_volume_signals"] = self.extract_advanced_price_volume_signals(symbol)
        
        print("  - Option chain signals...")
        compiled["option_chain_signals"] = self.extract_option_chain_signals(symbol)
        
        print("  - Futures data signals...")
        compiled["futures_signals"] = self.extract_futures_signals(symbol)
        
        print("  - Results comparison signals...")
        compiled["fundamental_signals"] = self.extract_results_signals(symbol)
        
        print("  - Historical stats signals...")
        compiled["historical_stats_signals"] = self.extract_historical_stats_signals(symbol)
        
        print("  - Stock details (analyst, risk, news, shareholding)...")
        compiled["stock_details_signals"] = self.extract_stock_details_signals(symbol)
        
        print("  - Sector signals...")
        compiled["sector_signals"] = self.extract_sector_signals(symbol)
        
        print("  - Intraday sector signals...")
        compiled["intraday_sector_signals"] = self.extract_intraday_sector_signals(symbol)
        
        print("  - Market status signals (Layer 3)...")
        compiled["market_status_signals"] = self.extract_market_status_signals(symbol)
        
        print("  - Index intraday signals (Layer 3)...")
        compiled["index_intraday_signals"] = self.extract_index_intraday_signals(symbol)
        
        print("  - FII/DII institutional flow signals...")
        compiled["fii_dii_signals"] = self.extract_fii_dii_signals(symbol)
        
        print("  - Index futures signals...")
        compiled["index_futures_signals"] = self.extract_index_futures_signals(symbol)
        
        print("  - Index history signals...")
        compiled["index_history_signals"] = self.extract_index_history_signals(symbol)
        
        print("  - Index option chain signals...")
        compiled["index_option_chain_signals"] = self.extract_index_option_chain_signals(symbol)
        
        print("  - VIX history signals...")
        compiled["vix_history_signals"] = self.extract_vix_history_signals(symbol)
        
        print("  - Block/Bulk deals signals...")
        compiled["block_bulk_deals_signals"] = self.extract_block_bulk_deals_signals(symbol)
        
        # Generate summary narrative
        compiled["narrative_summary"] = self._generate_narrative(compiled)
        
        return compiled
    
    def _generate_narrative(self, compiled: Dict) -> Dict[str, Any]:
        """
        Generate timeframe-structured narrative summary for LLM consumption.
        Organizes signals into three trading horizons: Intraday, Swing, Positional.
        """
        
        # ===============================================================
        # INTRADAY VIEW (0-1 day trading horizon)
        # Focus: Real-time price action, VWAP, market microstructure,
        #        immediate momentum, index intraday, VIX spikes
        # ===============================================================
        
        intraday_view = {
            "timeframe": "0-1 Day",
            "trading_style": "Scalping / Day Trading",
            "key_focus": "Price action, VWAP, momentum, market structure",
            "signals": {},
            "composite_factors": {"bullish": 0, "bearish": 0, "total": 0},
            "bias": "Neutral",
            "summary": ""
        }
        
        intra_bullish = 0
        intra_bearish = 0
        intra_signals = {}
        
        # Intraday stock signals
        intra = compiled.get("intraday_signals", {})
        if intra.get("available"):
            pa = intra.get("price_action", {})
            vol = intra.get("volatility", {})
            mom = intra.get("momentum", {})
            sess = intra.get("session_insights", {})
            strat = intra.get("strategy_inputs", {})
            
            intra_signals["price_action"] = {
                "current_price": pa.get("current_price"),
                "vwap": pa.get("vwap"),
                "vwap_gap_pct": pa.get("vwap_gap_pct"),
                "vwap_signal": pa.get("vwap_signal"),
                "intraday_trend": pa.get("intraday_trend"),
                "intraday_high": pa.get("intraday_high"),
                "intraday_low": pa.get("intraday_low"),
                "opening_range": pa.get("opening_range")
            }
            
            intra_signals["volatility"] = {
                "realized_vol": vol.get("realized_intraday_vol"),
                "volatility_spikes": vol.get("volatility_spikes"),
                "range_expansion": vol.get("range_expansion"),
                "status": vol.get("volatility_status")
            }
            
            intra_signals["momentum"] = {
                "slope_momentum": mom.get("slope_momentum"),
                "rate_of_change": mom.get("rate_of_change"),
                "mean_reversion_signal": mom.get("mean_reversion_signal"),
                "momentum_bias": mom.get("momentum_bias")
            }
            
            intra_signals["session"] = {
                "pattern": sess.get("session_pattern"),
                "trend_continuation": sess.get("trend_continuation")
            }
            
            intra_signals["scalp_params"] = {
                "scalp_range": strat.get("scalp_range"),
                "stop_loss_suggestion": strat.get("stop_loss_suggestion")
            }
            
            # Scoring
            if pa.get("vwap_signal") == "Bullish": intra_bullish += 1
            elif pa.get("vwap_signal") == "Bearish": intra_bearish += 1
            if pa.get("intraday_trend") == "Uptrend": intra_bullish += 1
            elif pa.get("intraday_trend") == "Downtrend": intra_bearish += 1
            if mom.get("momentum_bias") == "Bullish": intra_bullish += 1
            elif mom.get("momentum_bias") == "Bearish": intra_bearish += 1
            if sess.get("trend_continuation"): intra_bullish += 1
            else: intra_bearish += 1
        
        # Intraday sector signals
        isec = compiled.get("intraday_sector_signals", {})
        if isec.get("available"):
            sit = isec.get("sector_intraday_trend", {})
            vb = isec.get("vwap_behavior", {})
            ma = isec.get("momentum_analysis", {})
            irs = isec.get("intraday_relative_strength", {})
            ts = isec.get("trading_signals", {})
            cis = isec.get("composite_intraday_signal", {})
            
            intra_signals["sector_intraday"] = {
                "trend": sit.get("trend"),
                "change_pct": sit.get("change_pct"),
                "range_position_pct": sit.get("range_position_pct"),
                "vwap_position": vb.get("price_vs_vwap"),
                "vwap_dominance": vb.get("vwap_dominance"),
                "momentum_phase": ma.get("momentum_phase"),
                "momentum_direction": ma.get("momentum_direction"),
                "stock_vs_sector_alpha": irs.get("relative_strength_pct"),
                "rs_signal": irs.get("rs_signal"),
                "entry_timing": ts.get("entry_timing"),
                "breakout_quality": ts.get("breakout_quality"),
                "sector_bias": cis.get("intraday_bias")
            }
            
            if "Bullish" in cis.get("intraday_bias", ""): intra_bullish += 1
            elif "Bearish" in cis.get("intraday_bias", ""): intra_bearish += 1
            if "Favorable Long" in ts.get("entry_timing", ""): intra_bullish += 1
            elif "Favorable Short" in ts.get("entry_timing", ""): intra_bearish += 1
        
        # Index intraday signals
        idx = compiled.get("index_intraday_signals", {})
        if idx.get("available"):
            itm = idx.get("index_trend_momentum", {})
            vr = idx.get("volatility_regime", {})
            kl = idx.get("key_levels", {})
            ms = idx.get("market_structure", {})
            istr = idx.get("index_strength", {})
            rs = idx.get("relative_strength", {})
            ba = idx.get("beta_analysis", {})
            ts = idx.get("trading_signals", {})
            cis = idx.get("composite_index_signal", {})
            
            intra_signals["index_intraday"] = {
                "nifty_trend": itm.get("trend"),
                "nifty_change_pct": itm.get("change_pct"),
                "momentum_shift": itm.get("momentum_shift"),
                "day_type": vr.get("day_type"),
                "volatility_status": vr.get("volatility_status"),
                "opening_range_position": kl.get("opening_range_position"),
                "market_structure": ms.get("structure"),
                "breakout_status": ms.get("breakout_status"),
                "strength_score": istr.get("strength_score"),
                "strength_rating": istr.get("strength_rating"),
                "stock_alpha_pct": rs.get("alpha_pct"),
                "direction_signal": rs.get("direction_signal"),
                "intraday_beta": ba.get("intraday_beta"),
                "beta_signal": ba.get("beta_signal"),
                "long_signal": ts.get("long_signal"),
                "short_signal": ts.get("short_signal"),
                "signal_quality": ts.get("signal_quality"),
                "index_bias": cis.get("index_bias")
            }
            
            if "Bullish" in cis.get("index_bias", ""): intra_bullish += 1
            elif "Bearish" in cis.get("index_bias", ""): intra_bearish += 1
            if itm.get("trend") == "Bullish": intra_bullish += 1
            elif itm.get("trend") == "Bearish": intra_bearish += 1
            if "Valid Long" in ts.get("long_signal", ""): intra_bullish += 1
            if "Valid Short" in ts.get("short_signal", ""): intra_bearish += 1
        
        # Market status (GIFT NIFTY, gap)
        mkt = compiled.get("market_status_signals", {})
        if mkt.get("available"):
            gn = mkt.get("gift_nifty", {})
            irt = mkt.get("index_risk_tone", {})
            
            intra_signals["market_open"] = {
                "gift_nifty_price": gn.get("last_price"),
                "gift_nifty_change_pct": gn.get("pct_change"),
                "expected_gap_pct": gn.get("expected_gap_pct"),
                "overnight_sentiment": gn.get("overnight_sentiment"),
                "global_cue_strength": gn.get("global_cue_strength"),
                "risk_tone": irt.get("risk_tone")
            }
            
            if gn.get("expected_gap_pct", 0) > 0.1: intra_bullish += 1
            elif gn.get("expected_gap_pct", 0) < -0.1: intra_bearish += 1
            if "Risk-On" in irt.get("risk_tone", ""): intra_bullish += 1
            elif "Risk-Off" in irt.get("risk_tone", ""): intra_bearish += 1
        
        # VIX for intraday volatility context
        vix = compiled.get("vix_history_signals", {})
        if vix.get("available"):
            vr = vix.get("volatility_regime", {})
            vs = vix.get("vix_spikes", {})
            eqc = vix.get("equity_correlation", {})
            
            intra_signals["volatility_context"] = {
                "current_vix": vr.get("current_vix"),
                "vix_regime": vr.get("regime"),
                "market_mode": vr.get("market_mode"),
                "today_vix_change_pct": vs.get("today_change_pct"),
                "spike_signal": vs.get("spike_signal"),
                "entry_timing": eqc.get("entry_timing") if eqc else None
            }
            
            if vr.get("current_vix", 20) < 15: intra_bullish += 1
            elif vr.get("current_vix", 20) > 25: intra_bearish += 1
        
        # Options IV for intraday expected move
        oc = compiled.get("option_chain_signals", {})
        if oc.get("available"):
            vr = oc.get("volatility_risk", {})
            intra_signals["expected_move"] = {
                "atm_iv": vr.get("atm_iv"),
                "expected_move_pct": vr.get("expected_move_pct"),
                "iv_signal": vr.get("iv_signal")
            }
        
        # Intraday composite
        intra_total = intra_bullish + intra_bearish
        intraday_view["signals"] = intra_signals
        intraday_view["composite_factors"] = {"bullish": intra_bullish, "bearish": intra_bearish, "total": intra_total}
        
        if intra_bullish >= intra_bearish + 3:
            intraday_view["bias"] = "Strong Bullish"
        elif intra_bullish > intra_bearish:
            intraday_view["bias"] = "Bullish"
        elif intra_bearish >= intra_bullish + 3:
            intraday_view["bias"] = "Strong Bearish"
        elif intra_bearish > intra_bullish:
            intraday_view["bias"] = "Bearish"
        else:
            intraday_view["bias"] = "Neutral"
        
        intraday_view["summary"] = f"Intraday Bias: {intraday_view['bias']} ({intra_bullish} bullish / {intra_bearish} bearish factors). "
        if intra_signals.get("price_action"):
            intraday_view["summary"] += f"VWAP: {intra_signals['price_action'].get('vwap_signal', 'N/A')}. "
        if intra_signals.get("index_intraday"):
            intraday_view["summary"] += f"NIFTY: {intra_signals['index_intraday'].get('nifty_trend', 'N/A')}. "
        
        # ===============================================================
        # SWING VIEW (2-14 days trading horizon)
        # Focus: Short-term trend, RSI/MACD, option chain levels,
        #        futures OI, sector strength, FII/DII daily flows
        # ===============================================================
        
        swing_view = {
            "timeframe": "2-14 Days",
            "trading_style": "Swing Trading",
            "key_focus": "Short-term trend, momentum oscillators, F&O positioning, sector rotation",
            "signals": {},
            "composite_factors": {"bullish": 0, "bearish": 0, "total": 0},
            "bias": "Neutral",
            "summary": ""
        }
        
        swing_bullish = 0
        swing_bearish = 0
        swing_signals = {}
        
        # Historical signals - short term focus
        hist = compiled.get("historical_signals", {})
        if hist.get("available"):
            ts = hist.get("trend_structure", {})
            rv = hist.get("returns_volatility", {})
            vp = hist.get("volume_participation", {})
            ss = hist.get("strategy_signals", {})
            kl = hist.get("key_levels", {})
            
            swing_signals["trend"] = {
                "short_term_trend": ts.get("trend"),
                "price_structure": ts.get("price_structure"),
                "sma_20": ts.get("sma_20"),
                "sma_50": ts.get("sma_50"),
                "current_vs_sma20": "Above" if (hist.get("current_raw_price", 0) > ts.get("sma_20", 0)) else "Below",
                "higher_highs_20d": ts.get("higher_highs_count_20d"),
                "lower_lows_20d": ts.get("lower_lows_count_20d"),
                "pct_from_52_high": ts.get("pct_from_52_high")
            }
            
            swing_signals["momentum"] = {
                "rsi_14": ss.get("rsi_14"),
                "rsi_signal": ss.get("rsi_signal"),
                "weekly_return_pct": rv.get("weekly_return_pct"),
                "monthly_return_pct": rv.get("monthly_return_pct"),
                "atr_14": rv.get("atr_14"),
                "risk_level": rv.get("risk_level")
            }
            
            swing_signals["volume"] = {
                "volume_ratio": vp.get("volume_ratio"),
                "volume_spikes_20d": vp.get("volume_spikes_20d"),
                "avg_delivery_pct": vp.get("avg_delivery_pct"),
                "current_delivery_pct": vp.get("current_delivery_pct"),
                "price_volume_signal": vp.get("price_volume_signal"),
                "accumulation_signal": vp.get("accumulation_signal")
            }
            
            swing_signals["key_levels"] = {
                "pivot": kl.get("pivot"),
                "resistance_1": kl.get("resistance_1"),
                "resistance_2": kl.get("resistance_2"),
                "support_1": kl.get("support_1"),
                "support_2": kl.get("support_2")
            }
            
            swing_signals["stop_target"] = {
                "stop_loss_atr": ss.get("stop_loss_atr"),
                "target_atr": ss.get("target_atr")
            }
            
            # Scoring
            if "Uptrend" in ts.get("trend", ""): swing_bullish += 1
            elif "Downtrend" in ts.get("trend", ""): swing_bearish += 1
            if ss.get("rsi_14", 50) > 55: swing_bullish += 1
            elif ss.get("rsi_14", 50) < 45: swing_bearish += 1
            if "Accumulation" in vp.get("accumulation_signal", ""): swing_bullish += 1
            elif "Distribution" in vp.get("accumulation_signal", ""): swing_bearish += 1
        
        # Price-volume advanced signals
        pv = compiled.get("price_volume_signals", {})
        if pv.get("available"):
            pa = pv.get("price_action", {})
            vi = pv.get("volume_intelligence", {})
            sm = pv.get("smart_money_signals", {})
            ti = pv.get("technical_indicators", {})
            cs = pv.get("composite_summary", {})
            
            swing_signals["technical_indicators"] = {
                "ema_trend": pa.get("ema_trend"),
                "price_structure": pa.get("price_structure"),
                "breakout_signal": pa.get("breakout_signal"),
                "volatility_state": pa.get("volatility_state"),
                "macd_line": ti.get("macd_line"),
                "macd_signal": ti.get("macd_signal"),
                "macd_histogram": ti.get("macd_histogram"),
                "macd_crossover": ti.get("macd_crossover"),
                "macd_momentum": ti.get("macd_momentum"),
                "rsi_14": ti.get("rsi_14"),
                "rsi_divergence": ti.get("rsi_divergence")
            }
            
            swing_signals["smart_money"] = {
                "signal": sm.get("smart_money_signal"),
                "delivery_vs_avg": sm.get("delivery_vs_avg"),
                "delivery_trend": sm.get("delivery_trend"),
                "ad_trend": sm.get("ad_trend"),
                "institutional_activity": sm.get("institutional_activity")
            }
            
            swing_signals["volume_intelligence"] = {
                "volume_trend": vi.get("volume_trend"),
                "price_volume_confirmation": vi.get("price_volume_confirmation"),
                "climax_signal": vi.get("climax_signal"),
                "obv_trend": vi.get("obv_trend"),
                "obv_divergence": vi.get("obv_divergence")
            }
            
            swing_signals["composite_technical"] = {
                "bullish_signals": cs.get("bullish_signals"),
                "bearish_signals": cs.get("bearish_signals"),
                "signal_ratio": cs.get("signal_ratio"),
                "composite_bias": cs.get("composite_bias")
            }
            
            # Scoring
            if "Bullish" in cs.get("composite_bias", ""): swing_bullish += 1
            elif "Bearish" in cs.get("composite_bias", ""): swing_bearish += 1
            if "Accumulation" in sm.get("smart_money_signal", ""): swing_bullish += 1
            elif "Distribution" in sm.get("smart_money_signal", ""): swing_bearish += 1
            if ti.get("macd_momentum") == "Bullish": swing_bullish += 1
            elif ti.get("macd_momentum") == "Bearish": swing_bearish += 1
            if ti.get("macd_crossover") == "Bullish Crossover": swing_bullish += 1
            elif ti.get("macd_crossover") == "Bearish Crossover": swing_bearish += 1
        
        # Stock option chain for swing
        oc = compiled.get("option_chain_signals", {})
        if oc.get("available"):
            me = oc.get("market_expectation", {})
            ol = oc.get("options_levels", {})
            cos = oc.get("composite_options_signal", {})
            
            swing_signals["options"] = {
                "pcr_oi": me.get("pcr_oi"),
                "pcr_signal": me.get("pcr_signal"),
                "max_pain_strike": me.get("max_pain_strike"),
                "put_support_1": ol.get("put_support_1", {}).get("strike"),
                "put_support_1_oi": ol.get("put_support_1", {}).get("oi"),
                "call_resistance_1": ol.get("call_resistance_1", {}).get("strike"),
                "call_resistance_1_oi": ol.get("call_resistance_1", {}).get("oi"),
                "options_bias": cos.get("options_bias")
            }
            
            if "Bullish" in cos.get("options_bias", ""): swing_bullish += 1
            elif "Bearish" in cos.get("options_bias", ""): swing_bearish += 1
            if me.get("pcr_oi", 0) > 1.0: swing_bullish += 1
            elif me.get("pcr_oi", 0) < 0.7: swing_bearish += 1
        
        # Stock futures for swing
        fut = compiled.get("futures_signals", {})
        if fut.get("available"):
            td = fut.get("trend_direction", {})
            ps = fut.get("positioning_sentiment", {})
            cfs = fut.get("composite_futures_signal", {})
            
            swing_signals["futures"] = {
                "futures_premium_pct": td.get("futures_premium_pct"),
                "premium_signal": td.get("premium_signal"),
                "rollover_pct": td.get("rollover_pct"),
                "oi_signal": ps.get("oi_signal"),
                "today_oi_change_pct": ps.get("today_oi_change_pct"),
                "cot_proxy": ps.get("cot_proxy"),
                "futures_bias": cfs.get("futures_bias")
            }
            
            if "Bullish" in cfs.get("futures_bias", ""): swing_bullish += 1
            elif "Bearish" in cfs.get("futures_bias", ""): swing_bearish += 1
            if "Long Buildup" in ps.get("oi_signal", ""): swing_bullish += 1
            elif "Short Buildup" in ps.get("oi_signal", ""): swing_bearish += 1
        
        # Sector signals for swing
        sec = compiled.get("sector_signals", {})
        if sec.get("available"):
            stm = sec.get("sector_trend_momentum", {})
            rs = sec.get("relative_strength", {})
            cd = sec.get("confirmation_divergence", {})
            css = sec.get("composite_sector_signal", {})
            
            swing_signals["sector"] = {
                "sector_trend": stm.get("trend"),
                "sector_return_20d_pct": stm.get("return_20d_pct"),
                "stock_vs_sector_rs_pct": rs.get("relative_strength_pct"),
                "rs_signal": rs.get("rs_signal"),
                "trend_alignment": cd.get("trend_alignment"),
                "trade_conviction": cd.get("trade_conviction"),
                "sector_bias": css.get("sector_bias")
            }
            
            if "Bullish" in css.get("sector_bias", ""): swing_bullish += 1
            elif "Bearish" in css.get("sector_bias", ""): swing_bearish += 1
            if "Outperformer" in rs.get("rs_signal", ""): swing_bullish += 1
            elif "Underperformer" in rs.get("rs_signal", ""): swing_bearish += 1
        
        # FII/DII for swing
        fii = compiled.get("fii_dii_signals", {})
        if fii.get("available"):
            fb = fii.get("flow_bias", {})
            tp = fii.get("trend_persistence", {})
            cfs = fii.get("composite_fii_dii_signal", {})
            
            swing_signals["institutional_flows"] = {
                "fii_net_cr": fb.get("fii_net_cr"),
                "fii_stance": fb.get("fii_stance"),
                "dii_net_cr": fb.get("dii_net_cr"),
                "dii_stance": fb.get("dii_stance"),
                "total_net_cr": fb.get("total_net_cr"),
                "flow_trend": tp.get("flow_trend"),
                "institutional_bias": cfs.get("institutional_bias")
            }
            
            if "Bullish" in cfs.get("institutional_bias", ""): swing_bullish += 1
            elif "Bearish" in cfs.get("institutional_bias", ""): swing_bearish += 1
            if fb.get("total_net_cr", 0) > 500: swing_bullish += 1
            elif fb.get("total_net_cr", 0) < -500: swing_bearish += 1
        
        # Index futures for swing context
        idxf = compiled.get("index_futures_signals", {})
        if idxf.get("available"):
            poi = idxf.get("price_oi_interpretation", {})
            cif = idxf.get("composite_index_futures", {})
            
            swing_signals["index_futures"] = {
                "interpretation": poi.get("interpretation"),
                "signal": poi.get("signal"),
                "buildup_strength": poi.get("buildup_strength"),
                "index_futures_bias": cif.get("futures_bias"),
                "trade_recommendation": cif.get("trade_recommendation")
            }
            
            if "Bullish" in cif.get("futures_bias", ""): swing_bullish += 1
            elif "Bearish" in cif.get("futures_bias", ""): swing_bearish += 1
        
        # Index option chain for swing levels
        ioc = compiled.get("index_option_chain_signals", {})
        if ioc.get("available"):
            mp = ioc.get("max_pain", {})
            pcr = ioc.get("pcr_analysis", {})
            sr = ioc.get("support_resistance", {})
            cio = ioc.get("composite_index_options", {})
            
            sup_levels = sr.get("support_levels", [])
            res_levels = sr.get("resistance_levels", [])
            
            swing_signals["index_options"] = {
                "nifty_max_pain": mp.get("strike"),
                "max_pain_interpretation": mp.get("interpretation"),
                "nifty_oi_pcr": pcr.get("oi_pcr"),
                "nifty_pcr_signal": pcr.get("pcr_signal"),
                "nifty_support_1": sup_levels[0].get("strike") if sup_levels else None,
                "nifty_resistance_1": res_levels[0].get("strike") if res_levels else None,
                "nifty_options_bias": cio.get("options_bias")
            }
            
            if "Bullish" in cio.get("options_bias", ""): swing_bullish += 1
            elif "Bearish" in cio.get("options_bias", ""): swing_bearish += 1
        
        # Block/Bulk deals for swing
        bbd = compiled.get("block_bulk_deals_signals", {})
        if bbd.get("available"):
            ia = bbd.get("institutional_activity", {})
            cds = bbd.get("composite_deals_signal", {})
            
            swing_signals["deals"] = {
                "total_deals": bbd.get("total_deals"),
                "net_value_cr": ia.get("net_value_cr"),
                "activity_signal": ia.get("activity_signal"),
                "conviction": ia.get("conviction"),
                "deals_bias": cds.get("deals_bias")
            }
            
            if "Bullish" in cds.get("deals_bias", ""): swing_bullish += 1
            elif "Bearish" in cds.get("deals_bias", ""): swing_bearish += 1
        
        # Swing composite
        swing_total = swing_bullish + swing_bearish
        swing_view["signals"] = swing_signals
        swing_view["composite_factors"] = {"bullish": swing_bullish, "bearish": swing_bearish, "total": swing_total}
        
        if swing_bullish >= swing_bearish + 5:
            swing_view["bias"] = "Strong Bullish"
        elif swing_bullish > swing_bearish:
            swing_view["bias"] = "Bullish"
        elif swing_bearish >= swing_bullish + 5:
            swing_view["bias"] = "Strong Bearish"
        elif swing_bearish > swing_bullish:
            swing_view["bias"] = "Bearish"
        else:
            swing_view["bias"] = "Neutral"
        
        swing_view["summary"] = f"Swing Bias: {swing_view['bias']} ({swing_bullish} bullish / {swing_bearish} bearish factors). "
        if swing_signals.get("technical_indicators"):
            swing_view["summary"] += f"MACD: {swing_signals['technical_indicators'].get('macd_momentum', 'N/A')}. "
        if swing_signals.get("futures"):
            swing_view["summary"] += f"Futures: {swing_signals['futures'].get('oi_signal', 'N/A')}. "
        
        # ===============================================================
        # POSITIONAL VIEW (15+ days trading horizon)
        # Focus: Long-term trend (200 MA), fundamentals, growth metrics,
        #        shareholding, analyst ratings, index history
        # ===============================================================
        
        positional_view = {
            "timeframe": "15+ Days",
            "trading_style": "Positional / Investment",
            "key_focus": "Long-term trend, fundamentals, valuations, institutional ownership",
            "signals": {},
            "composite_factors": {"bullish": 0, "bearish": 0, "total": 0},
            "bias": "Neutral",
            "summary": ""
        }
        
        pos_bullish = 0
        pos_bearish = 0
        pos_signals = {}
        
        # Long-term trend from historical
        if hist.get("available"):
            ts = hist.get("trend_structure", {})
            rv = hist.get("returns_volatility", {})
            
            pos_signals["long_term_trend"] = {
                "sma_200": ts.get("sma_200"),
                "price_vs_sma200": "Above" if (hist.get("current_raw_price", 0) > ts.get("sma_200", 0)) else "Below",
                "current_vs_sma200_pct": ts.get("current_vs_sma200_pct"),
                "week_52_high": ts.get("week_52_high"),
                "week_52_low": ts.get("week_52_low"),
                "pct_from_52_high": ts.get("pct_from_52_high"),
                "near_ath": ts.get("near_ath"),
                "yearly_return_pct": rv.get("yearly_return_pct"),
                "max_drawdown_pct": rv.get("max_drawdown_pct")
            }
            
            if hist.get("current_raw_price", 0) > ts.get("sma_200", 0): pos_bullish += 1
            else: pos_bearish += 1
            if ts.get("near_ath"): pos_bullish += 1
            if rv.get("yearly_return_pct", 0) > 15: pos_bullish += 1
            elif rv.get("yearly_return_pct", 0) < 0: pos_bearish += 1
        
        # Fundamentals (from results_comparison)
        fund = compiled.get("fundamental_signals", {})
        # Stock details (for balance sheet, valuation, cash flow, etc.)
        stock = compiled.get("stock_details_signals", {})
        
        if fund.get("available"):
            ep = fund.get("earnings_profitability", {})
            
            pos_signals["earnings"] = {
                "net_profit_margin_pct": ep.get("net_profit_margin_pct"),
                "yoy_profit_growth_pct": ep.get("yoy_profit_growth_pct"),
                "qoq_profit_growth_pct": ep.get("qoq_profit_growth_pct"),
                "revenue_growth_yoy_pct": ep.get("yoy_revenue_growth_pct"),  # Corrected key name
                "eps_trend": ep.get("eps_trend"),
                "profitability_rating": ep.get("profitability_rating")
            }
            
            if ep.get("yoy_profit_growth_pct", 0) > 10: pos_bullish += 1
            elif ep.get("yoy_profit_growth_pct", 0) < -10: pos_bearish += 1
        
        # Use stock_details_signals for financial strength, valuation, etc.
        if stock.get("available"):
            fs = stock.get("financial_strength", {})
            vm = stock.get("valuation_metrics", {})
            gm = stock.get("growth_metrics", {})
            psd = stock.get("per_share_data", {})
            
            # Helper function to safely convert to float
            def safe_float(val, default=0.0):
                if val is None:
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            # Determine financial health based on ratios
            financial_health = "Average"
            current_ratio = safe_float(fs.get("current_ratio"), 0)
            debt_to_equity = safe_float(fs.get("debt_to_equity"), 1)
            interest_coverage = safe_float(fs.get("interest_coverage"), 0)
            
            if current_ratio > 1.5 and debt_to_equity < 0.5 and interest_coverage > 5:
                financial_health = "Strong"
            elif current_ratio < 1.0 or debt_to_equity > 1.5:
                financial_health = "Weak"
            
            pos_signals["balance_sheet"] = {
                "debt_to_equity": fs.get("debt_to_equity"),
                "current_ratio": fs.get("current_ratio"),
                "interest_coverage": fs.get("interest_coverage"),
                "financial_health": financial_health
            }
            
            # Determine valuation signal
            pe = vm.get("pe_ratio") or vm.get("pe_ttm")
            pb = vm.get("price_to_book")
            valuation_signal = "Fair"
            if pe and pb:
                if pe < 15 and pb < 2: valuation_signal = "Undervalued"
                elif pe > 35 or pb > 5: valuation_signal = "Overvalued"
            
            pos_signals["valuation"] = {
                "pe_ratio": pe,
                "pe_vs_sector": vm.get("pe_vs_sector", "N/A"),
                "pb_ratio": pb,
                "peg_ratio": vm.get("peg_ratio"),
                "valuation_signal": valuation_signal
            }
            
            # Cash flow from financial strength data
            fcf = fs.get("free_cash_flow")
            pos_signals["cash_flow"] = {
                "operating_cash_flow_cr": None,  # Not directly available
                "free_cash_flow_cr": round(fcf / 100, 2) if fcf else None,  # Convert to Cr
                "fcf_margin_pct": None,
                "cash_conversion": "Positive" if fcf and fcf > 0 else ("Negative" if fcf else None)
            }
            
            # Dividends
            div_yield = vm.get("dividend_yield") or vm.get("dividend_yield_5yr")
            payout = fs.get("payout_ratio")
            pos_signals["dividends"] = {
                "dividend_yield_pct": div_yield,
                "dividend_payout_pct": payout,
                "dividend_consistency": "Regular" if div_yield and div_yield > 0 else "None/Irregular"
            }
            
            # Growth quality from growth_metrics
            rev_growth = gm.get("revenue_growth_5yr") or gm.get("revenue_growth_rate")
            profit_growth = gm.get("eps_growth_5yr") or gm.get("earnings_growth")
            quality_score = "Average"
            if rev_growth and profit_growth:
                if rev_growth > 10 and profit_growth > 10: quality_score = "High"
                elif rev_growth < 0 or profit_growth < 0: quality_score = "Low"
            
            pos_signals["growth_quality"] = {
                "revenue_cagr_3yr": gm.get("revenue_growth_3yr"),
                "profit_cagr_3yr": gm.get("eps_growth_3yr"),
                "growth_consistency": gm.get("earnings_growth_consistency", "N/A"),
                "quality_score": quality_score
            }
            
            # Fundamental composite calculation
            fund_bullish = 0
            fund_bearish = 0
            if financial_health == "Strong": fund_bullish += 1
            elif financial_health == "Weak": fund_bearish += 1
            if valuation_signal == "Undervalued": fund_bullish += 1
            elif valuation_signal == "Overvalued": fund_bearish += 1
            if fcf and fcf > 0: fund_bullish += 1
            elif fcf and fcf < 0: fund_bearish += 1
            if quality_score == "High": fund_bullish += 1
            elif quality_score == "Low": fund_bearish += 1
            
            fund_bias = "Neutral"
            if fund_bullish >= fund_bearish + 2: fund_bias = "Bullish"
            elif fund_bearish >= fund_bullish + 2: fund_bias = "Bearish"
            
            pos_signals["fundamental_composite"] = {
                "fundamental_bias": fund_bias,
                "bullish_factors": fund_bullish,
                "bearish_factors": fund_bearish
            }
            
            if fund_bias == "Bullish": pos_bullish += 1
            elif fund_bias == "Bearish": pos_bearish += 1
            if financial_health == "Strong": pos_bullish += 1
            elif financial_health == "Weak": pos_bearish += 1
            
            # Analyst, shareholding, risk (continuing from stock_details_signals)
            analyst = stock.get("analyst_recommendations", {})
            risk = stock.get("risk_assessment", {})
            sh = stock.get("shareholding", {})
            ns = stock.get("news_sentiment", {})
            
            pos_signals["analyst"] = {
                "consensus": analyst.get("consensus"),
                "buy_pct": analyst.get("buy_pct"),
                "sell_pct": analyst.get("sell_pct"),
                "total_analysts": analyst.get("total_analysts"),
                "mean_rating": analyst.get("mean_rating")
            }
            
            pos_signals["risk"] = {
                "risk_category": risk.get("risk_category"),
                "std_deviation": risk.get("std_deviation"),
                "risk_level": risk.get("risk_level")
            }
            
            pos_signals["shareholding"] = {
                "fii": sh.get("fii", {}),
                "dii": sh.get("dii", {}),
                "mf": sh.get("mf", {}),
                "promoter": sh.get("promoter", {}),
                "institutional_sentiment": sh.get("institutional_sentiment")
            }
            
            pos_signals["growth_metrics"] = {
                "revenue_growth_5yr": gm.get("revenue_growth_5yr"),
                "eps_growth_5yr": gm.get("eps_growth_5yr"),
                "book_value_growth_5yr": gm.get("book_value_growth_5yr")
            }
            
            pos_signals["valuation_metrics"] = {
                "pe_ttm": vm.get("pe_ttm"),
                "peg_ratio": vm.get("peg_ratio"),
                "price_to_book": vm.get("price_to_book"),
                "dividend_yield": vm.get("dividend_yield"),
                "ev_commentary": vm.get("ev_commentary")
            }
            
            pos_signals["news_sentiment"] = {
                "overall": ns.get("overall"),
                "positive_count": ns.get("positive_count"),
                "negative_count": ns.get("negative_count")
            }
            
            if analyst.get("buy_pct", 0) > 70: pos_bullish += 1
            elif analyst.get("sell_pct", 0) > 30: pos_bearish += 1
            if "Buying" in sh.get("institutional_sentiment", ""): pos_bullish += 1
            elif "Selling" in sh.get("institutional_sentiment", ""): pos_bearish += 1
        
        # Index history for positional context
        idxh = compiled.get("index_history_signals", {})
        if idxh.get("available"):
            pt = idxh.get("primary_trend", {})
            eqc = idxh.get("equity_correlation", {})
            cih = idxh.get("composite_index_history", {})
            
            pos_signals["index_trend"] = {
                "nifty_short_term": pt.get("short_term"),
                "nifty_medium_term": pt.get("medium_term"),
                "nifty_price_structure": pt.get("price_structure"),
                "stock_alpha_20d": eqc.get("alpha_20d") if eqc else None,
                "relative_strength": eqc.get("relative_strength") if eqc else None,
                "signal_validation": eqc.get("signal_validation") if eqc else None,
                "index_bias": cih.get("index_bias"),
                "positional_recommendation": cih.get("positional_recommendation")
            }
            
            if "Bullish" in cih.get("index_bias", ""): pos_bullish += 1
            elif "Bearish" in cih.get("index_bias", ""): pos_bearish += 1
            if pt.get("medium_term") == "Bullish": pos_bullish += 1
            elif pt.get("medium_term") == "Bearish": pos_bearish += 1
        
        # VIX for positional volatility regime
        vix = compiled.get("vix_history_signals", {})
        if vix.get("available"):
            vr = vix.get("volatility_regime", {})
            mr = vix.get("mean_reversion", {})
            cvs = vix.get("composite_vix_signal", {})
            
            pos_signals["volatility_regime"] = {
                "current_vix": vr.get("current_vix"),
                "regime": vr.get("regime"),
                "market_mode": vr.get("market_mode"),
                "z_score": mr.get("z_score"),
                "mean_reversion_signal": mr.get("signal"),
                "vix_bias": cvs.get("vix_bias"),
                "market_implication": cvs.get("market_implication")
            }
            
            if "Bullish" in cvs.get("vix_bias", ""): pos_bullish += 1
            elif "Bearish" in cvs.get("vix_bias", ""): pos_bearish += 1
        
        # Market macro backdrop for positional
        mkt = compiled.get("market_status_signals", {})
        if mkt.get("available"):
            mb = mkt.get("macro_backdrop", {})
            acf = mkt.get("asset_class_flows", {})
            cms = mkt.get("composite_market_signal", {})
            
            pos_signals["macro"] = {
                "total_market_cap_tr_usd": mb.get("total_market_cap_tr_usd"),
                "market_phase": mb.get("market_phase"),
                "macro_bias": mb.get("macro_bias"),
                "usdinr_rate": acf.get("usdinr_rate"),
                "inr_status": acf.get("inr_status"),
                "currency_impact": acf.get("currency_impact"),
                "market_bias": cms.get("market_bias")
            }
            
            if "Bullish" in mb.get("macro_bias", ""): pos_bullish += 1
            elif "Bearish" in mb.get("macro_bias", ""): pos_bearish += 1
        
        # Corporate actions
        intro = compiled.get("introduction", {})
        corp = intro.get("corporate_info", {})
        if corp:
            pos_signals["corporate_events"] = {
                "recent_announcements": len(corp.get("announcements", [])),
                "upcoming_board_meetings": len(corp.get("board_meetings", [])),
                "recent_dividends": len(corp.get("dividends", []))
            }
        
        # Positional composite
        pos_total = pos_bullish + pos_bearish
        positional_view["signals"] = pos_signals
        positional_view["composite_factors"] = {"bullish": pos_bullish, "bearish": pos_bearish, "total": pos_total}
        
        if pos_bullish >= pos_bearish + 4:
            positional_view["bias"] = "Strong Bullish"
        elif pos_bullish > pos_bearish:
            positional_view["bias"] = "Bullish"
        elif pos_bearish >= pos_bullish + 4:
            positional_view["bias"] = "Strong Bearish"
        elif pos_bearish > pos_bullish:
            positional_view["bias"] = "Bearish"
        else:
            positional_view["bias"] = "Neutral"
        
        positional_view["summary"] = f"Positional Bias: {positional_view['bias']} ({pos_bullish} bullish / {pos_bearish} bearish factors). "
        if pos_signals.get("analyst"):
            positional_view["summary"] += f"Analyst: {pos_signals['analyst'].get('consensus', 'N/A')}. "
        if pos_signals.get("long_term_trend"):
            positional_view["summary"] += f"vs 200 SMA: {pos_signals['long_term_trend'].get('price_vs_sma200', 'N/A')}. "
        
        # ===============================================================
        # OVERALL COMPOSITE
        # ===============================================================
        
        total_bullish = intra_bullish + swing_bullish + pos_bullish
        total_bearish = intra_bearish + swing_bearish + pos_bearish
        total_factors = total_bullish + total_bearish
        
        if total_bullish >= total_bearish + 10:
            overall_bias = "Strong Buy - Multiple timeframe confirmations"
        elif total_bullish > total_bearish:
            overall_bias = "Buy - Majority signals positive"
        elif total_bearish >= total_bullish + 10:
            overall_bias = "Strong Sell - Multiple timeframe confirmations"
        elif total_bearish > total_bullish:
            overall_bias = "Sell - Majority signals negative"
        else:
            overall_bias = "Neutral - Mixed signals across timeframes"
        
        # ===============================================================
        # RETURN STRUCTURED NARRATIVE
        # ===============================================================
        
        return {
            "intraday_view": intraday_view,
            "swing_view": swing_view,
            "positional_view": positional_view,
            "overall_composite": {
                "total_bullish_factors": total_bullish,
                "total_bearish_factors": total_bearish,
                "total_factors_analyzed": total_factors,
                "overall_bias": overall_bias,
                "summary": f"Overall: {overall_bias}. Intraday={intraday_view['bias']}, Swing={swing_view['bias']}, Positional={positional_view['bias']}."
            }
        }
    
    def _convert_numpy(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def save_compiled_json(self, symbol: str, output_dir: str = ".") -> str:
        """Generate and save compiled JSON"""
        compiled = self.compile_signals(symbol)
        compiled = self._convert_numpy(compiled)
        
        filename = f"{output_dir}/{symbol.upper()}_compiled.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(compiled, f, indent=2, ensure_ascii=False)
        
        print(f"\nCompiled signals saved to: {filename}")
        return filename
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    import sys
    
    if len(sys.argv) < 2:
        db_path = input("Enter database path: ").strip()
    else:
        db_path = sys.argv[1]
    
    if len(sys.argv) < 3:
        symbol = input("Enter stock symbol: ").strip()
    else:
        symbol = sys.argv[2]
    
    generator = SignalGenerator(db_path)
    output_file = generator.save_compiled_json(symbol, ".")
    generator.close()
    
    print(f"\nDone! Output: {output_file}")


if __name__ == "__main__":
    main()
