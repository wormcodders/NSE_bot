#!/usr/bin/env python3
"""
Stock Signal Bot - Automated Buy/Sell Signal Scanner with Telegram Notifications

Enhanced Features:
- Telegram bot for buy/sell signal notifications
- SQLite database for persistent portfolio management
- /add and /remove commands for portfolio management
- /portfolio command to view holdings with recommendations
- /suggest command for on-demand ticker analysis
- HOLD signal when ticker doesn't meet buy/sell criteria
- Built-in Web Dashboard for live signals and depth analysis

Author: MiniMax Agent
"""

# Fix timezone issue for yfinance - MUST be set before importing yfinance
import os
os.environ['TZ'] = 'Asia/Kolkata'

import sys
import json
import sqlite3
import logging
import asyncio
import gc
import warnings
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Flask imports for built-in web dashboard
try:
    from flask import Flask, render_template, jsonify, request, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Web dashboard will be disabled.")

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from config import (
        TELEGRAM_BOT_TOKEN,
        ADMIN_IDS,
        SCHEDULED_TIMES,
        ANALYSIS_PERIOD_DAYS,
        RSI_WINDOW,
        RSI_BUY_THRESHOLD,
        RSI_SELL_THRESHOLD,
        RECENT_DAYS_LOOKBACK,
    )
except ImportError:
    print("Error: config.py not found. Please create config.py with TELEGRAM_BOT_TOKEN and ADMIN_IDS.")
    sys.exit(1)

# Import LLM Stock Recommender functionality
LLM_AVAILABLE = False
LLM_IMPORT_ERROR = None

try:
    import llm_stock_recommendation
    from llm_stock_recommendation import LLMStockRecommender
    # Verify the class exists and is accessible
    if not hasattr(llm_stock_recommendation, 'LLMStockRecommender'):
        raise ImportError("LLMStockRecommender class not found in llm_stock_recommendation module")
    LLM_AVAILABLE = True
    print("✓ LLMStockRecommender imported successfully")
except ImportError as e:
    LLM_IMPORT_ERROR = str(e)
    print(f"✗ Import Error: {e}")
    print("  LLM depth analysis will be disabled.")
except Exception as e:
    LLM_IMPORT_ERROR = str(e)
    print(f"✗ Unexpected Error during import: {e}")
    import traceback
    traceback.print_exc()
    print("  LLM depth analysis will be disabled.")

# Configure logging with rotation to prevent log file growth
from logging.handlers import RotatingFileHandler

# Create rotating file handler
file_handler = RotatingFileHandler(
    "bot.log",
    maxBytes=250*1024,  # 250KB
    backupCount=2
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        file_handler,
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Global state
subscribers_file = Path(__file__).parent / "subscribers.json"
tickers_file = Path(__file__).parent / "tickers.xlsx"
database_file = Path(__file__).parent / "bot.db"
test_mode = False

# Store application reference for handlers
_bot_application = None

# Web dashboard configuration
WEB_DATA_DIR = Path(__file__).parent / "web_data"
WEB_DATA_DIR.mkdir(exist_ok=True)
SIGNALS_FILE = WEB_DATA_DIR / "signals_history.json"
CURRENT_SIGNALS_FILE = WEB_DATA_DIR / "current_signals.json"

# Flask app initialization (lazy initialization)
_flask_app = None
_flask_thread = None


# ==================== DATABASE FUNCTIONS ====================

def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            created_at TEXT
        )
    ''')
    
    # Create portfolios table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            ticker TEXT,
            added_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, ticker)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def get_user_portfolio(user_id: str) -> List[str]:
    """Get all tickers in user's portfolio."""
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT ticker FROM portfolios WHERE user_id = ? ORDER BY added_at DESC',
        (user_id,)
    )
    tickers = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return tickers


def add_to_portfolio(user_id: str, username: str, ticker: str) -> Tuple[bool, str]:
    """Add a ticker to user's portfolio."""
    ticker = ticker.upper().strip()
    now = datetime.now().isoformat()
    
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    # Ensure user exists
    cursor.execute(
        'INSERT OR IGNORE INTO users (user_id, username, created_at) VALUES (?, ?, ?)',
        (user_id, username, now)
    )
    
    try:
        cursor.execute(
            'INSERT INTO portfolios (user_id, ticker, added_at) VALUES (?, ?, ?)',
            (user_id, ticker, now)
        )
        conn.commit()
        success = True
        message = f"✅ {ticker} added to your portfolio"
    except sqlite3.IntegrityError:
        success = False
        message = f"⚠️ {ticker} is already in your portfolio"
    
    conn.close()
    return success, message


def remove_from_portfolio(user_id: str, ticker: str) -> Tuple[bool, str]:
    """Remove a ticker from user's portfolio."""
    ticker = ticker.upper().strip()
    
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    cursor.execute(
        'DELETE FROM portfolios WHERE user_id = ? AND ticker = ?',
        (user_id, ticker)
    )
    
    if cursor.rowcount > 0:
        conn.commit()
        success = True
        message = f"🗑️ {ticker} removed from your portfolio"
    else:
        success = False
        message = f"❌ {ticker} is not in your portfolio"
    
    conn.close()
    return success, message


# ==================== ANALYSIS FUNCTIONS ====================

def analyze_ticker_signal(ticker: str, period_days: int = ANALYSIS_PERIOD_DAYS, use_hold: bool = False) -> Optional[dict]:
    """
    Analyze a ticker and return signal information.
    
    Args:
        ticker: Stock ticker symbol
        period_days: Number of days of historical data to fetch
        use_hold: If True, use HOLD for no signal; if False, use WAIT
    
    Returns:
        Dictionary with signal, price, RSI, MACD, etc.
        Returns None if analysis fails or ticker is invalid.
    """
    try:
        # Set end date to today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Download data with timezone fix applied
        df = download_stock_data(ticker, start_date, end_date)
        
        if df is None or df.empty or len(df) < 50:
            return None
        
        # Handle the Close column - it might be a MultiIndex
        close_prices = _extract_close_prices(df)
        
        if close_prices is None or len(close_prices) < 50:
            return None
        
        # Calculate MACD
        macd_calc = ta.trend.MACD(close_prices)
        macd_line = macd_calc.macd()
        signal_line = macd_calc.macd_signal()
        
        # Calculate RSI
        rsi_calc = ta.momentum.RSIIndicator(close_prices, window=RSI_WINDOW)
        rsi_values = rsi_calc.rsi()
        
        # Get values as arrays
        macd_arr = macd_line.dropna().values
        signal_arr = signal_line.dropna().values
        rsi_arr = rsi_values.dropna().values
        
        if len(macd_arr) < 10:
            return None
        
        # Determine signal - use HOLD if in portfolio, otherwise use WAIT
        signal = "HOLD" if use_hold else "WAIT"
        crossover_type = None
        
        # Look at recent days for crossover
        lookback = min(RECENT_DAYS_LOOKBACK, len(macd_arr))
        
        for i in range(1, lookback):
            idx = -i
            prev_idx = -i - 1
            
            # Buy signal: MACD crosses above signal with RSI >= threshold
            if (macd_arr[idx] > signal_arr[idx] and 
                macd_arr[prev_idx] <= signal_arr[prev_idx] and
                rsi_arr[idx] >= RSI_BUY_THRESHOLD):
                signal = "BUY"
                crossover_type = "bullish"
                break
            
            # Sell signal: MACD crosses below signal
            if (macd_arr[idx] < signal_arr[idx] and 
                macd_arr[prev_idx] >= signal_arr[prev_idx]):
                signal = "SELL"
                crossover_type = "bearish"
                break
        
        # Get current values from the last valid close price
        current_price = round(float(close_prices.iloc[-1]), 2)
        current_rsi = round(rsi_arr[-1], 2)
        current_macd = round(macd_arr[-1], 4)
        current_signal = round(signal_arr[-1], 4)
        macd_histogram = round(current_macd - current_signal, 4)
        
        return {
            "ticker": ticker.upper(),
            "signal": signal,
            "signal_emoji": get_signal_emoji(signal),
            "current_price": current_price,
            "current_rsi": current_rsi,
            "current_macd": current_macd,
            "current_signal": current_signal,
            "macd_histogram": macd_histogram,
            "crossover_type": crossover_type,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        # Silently skip problematic tickers
        return None
    finally:
        # Explicitly cleanup large data structures
        try:
            del df, close_prices, macd_arr, signal_arr, rsi_arr
        except:
            pass


def download_stock_data(ticker: str, start_date: datetime, end_date: datetime):
    """
    Download stock data for analysis.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        DataFrame with stock data or None if download fails
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            threads=False
        )
    return df


def _extract_close_prices(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Extract close prices from DataFrame, handling MultiIndex and timezone issues.
    """
    try:
        # Handle the Close column - it might be a MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # If MultiIndex, try to find Close column
            if ('Close', '') in df.columns:
                close_prices = df['Close', '']
            elif ('Close', 'Close') in df.columns:
                close_prices = df['Close', 'Close']
            else:
                # Try first column that contains 'Close'
                for col in df.columns:
                    if 'Close' in str(col):
                        close_prices = df[col]
                        break
                else:
                    return None
        else:
            close_prices = df['Close']
        
        # Convert to Series and handle timezone
        if hasattr(close_prices, 'dt'):
            # If it's timezone-aware, convert to naive
            try:
                if close_prices.dt.tz is not None:
                    close_prices = close_prices.dt.tz_localize(None)
                elif hasattr(close_prices, 'dt'):
                    # Try converting with tz_localize(None) if it has timezone info
                    close_prices = pd.to_datetime(close_prices).tz_localize(None)
            except:
                pass
        
        # Convert to numpy array and back to Series to remove timezone
        close_prices = pd.Series(close_prices.values)
        
        # Drop NaN values
        close_prices = close_prices.dropna()
        
        return close_prices
        
    except Exception:
        return None


def get_signal_emoji(signal: str) -> str:
    """Get emoji for signal type."""
    emojis = {
        "BUY": "🟢",
        "SELL": "🔴",
        "HOLD": "🟡",
        "WAIT": "🟡",
    }
    return emojis.get(signal, "⚪")


def load_tickers() -> List[str]:
    """Load tickers from Excel file."""
    if not tickers_file.exists():
        logger.error(f"Tickers file not found: {tickers_file}")
        return []
    
    try:
        df = pd.read_excel(tickers_file)
        tickers = df.iloc[:, 0].dropna().astype(str).unique().tolist()
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        logger.info(f"Loaded {len(tickers)} tickers from Excel file")
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        return []


# ==================== WEB DASHBOARD FUNCTIONS ====================

def save_signals_for_web(
    buy_signals: List[dict], 
    sell_signals: List[dict], 
    summary: dict,
    scheduled_time: str = None
) -> bool:
    """
    Save signals to JSON file for web dashboard display.
    Only saves market signals (buy/sell), NOT portfolio info.
    
    Args:
        buy_signals: List of buy signal dictionaries
        sell_signals: List of sell signal dictionaries
        summary: Analysis summary dictionary
        scheduled_time: The scheduled time when analysis ran (e.g., "9:30")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get current time
        timestamp = datetime.now().isoformat()
        
        # Format time for display
        if not scheduled_time:
            scheduled_time = datetime.now().strftime("%H:%M")
        
        # Create clean signal data (no portfolio info)
        signal_data = {
            "timestamp": timestamp,
            "scheduled_time": scheduled_time,
            "market_scan": {
                "total_tickers": summary.get("total_tickers", 0),
                "analyzed": summary.get("analyzed", 0),
                "errors": summary.get("errors", 0),
                "execution_time": summary.get("execution_time", 0)
            },
            "buy_signals": [
                {
                    "ticker": s.get("ticker", ""),
                    "signal": s.get("signal", "BUY"),
                    "price": s.get("current_price", 0),
                    "rsi": s.get("current_rsi", 0),
                    "macd": s.get("current_macd", 0),
                    "signal_line": s.get("current_signal", 0),
                    "histogram": s.get("macd_histogram", 0),
                    "crossover_type": s.get("crossover_type", None)
                }
                for s in buy_signals
            ],
            "sell_signals": [
                {
                    "ticker": s.get("ticker", ""),
                    "signal": s.get("signal", "SELL"),
                    "price": s.get("current_price", 0),
                    "rsi": s.get("current_rsi", 0),
                    "macd": s.get("current_macd", 0),
                    "signal_line": s.get("current_signal", 0),
                    "histogram": s.get("macd_histogram", 0),
                    "crossover_type": s.get("crossover_type", None)
                }
                for s in sell_signals
            ]
        }
        
        # Save to current signals file
        with open(CURRENT_SIGNALS_FILE, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, indent=2, ensure_ascii=False)
        
        # Also update history
        _update_signals_history(signal_data)
        
        logger.info(f"Signals saved for web: {len(buy_signals)} buy, {len(sell_signals)} sell")
        return True
        
    except Exception as e:
        logger.error(f"Error saving signals for web: {e}")
        return False


def _update_signals_history(new_signal_data: dict) -> None:
    """Update the signals history file with new data."""
    try:
        history = {"signals_history": [], "last_updated": None}
        
        # Load existing history if exists
        if SIGNALS_FILE.exists():
            with open(SIGNALS_FILE, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = {"signals_history": [], "last_updated": None}
        
        # Add new signal data
        history["signals_history"].insert(0, new_signal_data)
        
        # Keep only last 100 entries
        history["signals_history"] = history["signals_history"][:100]
        
        # Update timestamp
        history["last_updated"] = datetime.now().isoformat()
        
        # Save
        with open(SIGNALS_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error updating signals history: {e}")


def get_current_signals() -> dict:
    """
    Get current signals for web dashboard.
    
    Returns:
        Dictionary with current signal data or empty structure
    """
    try:
        if CURRENT_SIGNALS_FILE.exists():
            with open(CURRENT_SIGNALS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading current signals: {e}")
    
    # Return empty structure
    return {
        "timestamp": None,
        "scheduled_time": None,
        "market_scan": {
            "total_tickers": 0,
            "analyzed": 0,
            "errors": 0,
            "execution_time": 0
        },
        "buy_signals": [],
        "sell_signals": []
    }


def get_signals_history(limit: int = 20) -> list:
    """
    Get historical signals for the web dashboard.
    
    Args:
        limit: Maximum number of history entries to return
    
    Returns:
        List of historical signal entries
    """
    try:
        if SIGNALS_FILE.exists():
            with open(SIGNALS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("signals_history", [])[:limit]
    except Exception as e:
        logger.error(f"Error reading signals history: {e}")
    
    return []


def format_signals_for_web(buy_signals: List[dict], sell_signals: List[dict]) -> str:
    """
    Format signals as HTML for web display.
    
    Args:
        buy_signals: List of buy signal dictionaries
        sell_signals: List of sell signal dictionaries
    
    Returns:
        HTML string for displaying signals
    """
    html_parts = []
    
    # Buy Signals Section
    if buy_signals:
        html_parts.append('<div class="signal-section">')
        html_parts.append('<h3 class="section-title buy-title">🟢 BUY SIGNALS</h3>')
        html_parts.append('<div class="signal-grid">')
        
        for signal in sorted(buy_signals, key=lambda x: x.get('ticker', '')):
            html_parts.append(f'''
                <div class="signal-card buy">
                    <div class="signal-header">
                        <span class="ticker">{signal.get('ticker', '')}</span>
                        <span class="signal-badge buy">BUY</span>
                    </div>
                    <div class="signal-price">₹{signal.get('price', 0):.2f}</div>
                    <div class="signal-indicators">
                        <span class="indicator">RSI: {signal.get('rsi', 0):.1f}</span>
                        <span class="indicator">MACD: {signal.get('histogram', 0):+.4f}</span>
                    </div>
                </div>
            ''')
        
        html_parts.append('</div></div>')
    
    # Sell Signals Section
    if sell_signals:
        html_parts.append('<div class="signal-section">')
        html_parts.append('<h3 class="section-title sell-title">🔴 SELL SIGNALS</h3>')
        html_parts.append('<div class="signal-grid">')
        
        for signal in sorted(sell_signals, key=lambda x: x.get('ticker', '')):
            html_parts.append(f'''
                <div class="signal-card sell">
                    <div class="signal-header">
                        <span class="ticker">{signal.get('ticker', '')}</span>
                        <span class="signal-badge sell">SELL</span>
                    </div>
                    <div class="signal-price">₹{signal.get('price', 0):.2f}</div>
                    <div class="signal-indicators">
                        <span class="indicator">RSI: {signal.get('rsi', 0):.1f}</span>
                        <span class="indicator">MACD: {signal.get('histogram', 0):+.4f}</span>
                    </div>
                </div>
            ''')
        
        html_parts.append('</div></div>')
    
    # No Signals
    if not buy_signals and not sell_signals:
        html_parts.append('''
            <div class="no-signals">
                <p>No buy or sell signals generated in this scan.</p>
                <p>The market may be in a consolidation phase.</p>
            </div>
        ''')
    
    return '\n'.join(html_parts)


def run_analysis() -> Tuple[List[dict], List[dict], dict]:
    """Run analysis on all tickers from Excel file."""
    global test_mode
    
    tickers = load_tickers()
    buy_signals = []
    sell_signals = []
    summary = {
        "total_tickers": len(tickers),
        "analyzed": 0,
        "errors": 0,
        "buy_count": 0,
        "sell_count": 0,
        "execution_time": None,
    }
    
    start_time = datetime.now()
    logger.info(f"Starting analysis on {len(tickers)} tickers")
    
    for ticker in tickers:
        result = analyze_ticker_signal(ticker)
        
        if result is None:
            summary["errors"] += 1
            continue
        
        summary["analyzed"] += 1
        
        if result["signal"] == "BUY":
            buy_signals.append(result)
            summary["buy_count"] += 1
            logger.info(f"BUY signal: {ticker}")
        
        if result["signal"] == "SELL":
            sell_signals.append(result)
            summary["sell_count"] += 1
            logger.info(f"SELL signal: {ticker}")
    
    summary["execution_time"] = (datetime.now() - start_time).total_seconds()
    logger.info(f"Analysis complete: {summary}")
    
    # Force garbage collection to prevent memory buildup
    gc.collect()
    
    return buy_signals, sell_signals, summary


# ==================== MESSAGE FORMATTING ====================

def format_signal_message(
    buy_signals: List[dict], 
    sell_signals: List[dict], 
    summary: dict,
    is_test: bool = False
) -> str:
    """Format signals into a Telegram message."""
    lines = []
    
    # Header
    if is_test:
        lines.append("🧪 *TEST MODE* 🧪")
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append(f"⚠️ *This is a test message - NOT a live signal*")
        lines.append("")
    else:
        lines.append("📊 *Market Scan Complete*")
        lines.append("━━━━━━━━━━━━━━━━")
    
    # Summary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"🕐 Scan Time: {timestamp}")
    lines.append(f"📈 Tickers Analyzed: {summary['analyzed']}")
    lines.append(f"❌ Errors: {summary['errors']}")
    lines.append(f"⏱️ Execution Time: {summary['execution_time']:.2f}s")
    lines.append("")
    
    # Buy Signals - Sort alphabetically by ticker
    if buy_signals:
        buy_signals_sorted = sorted(buy_signals, key=lambda x: x['ticker'])
        lines.append("🟢 *BUY SIGNALS*")
        lines.append("━━━━━━━━━━━━━━━━")
        for signal in buy_signals_sorted:
            lines.append(f"• *{signal['ticker']}* - ₹{signal['current_price']:.2f}")
            lines.append(f"  RSI: {signal['current_rsi']:.1f}")
        lines.append("")
    else:
        lines.append("🟢 *BUY SIGNALS*: None")
        lines.append("")
    
    # Sell Signals - Sort alphabetically by ticker
    if sell_signals:
        sell_signals_sorted = sorted(sell_signals, key=lambda x: x['ticker'])
        lines.append("🔴 *SELL SIGNALS*")
        lines.append("━━━━━━━━━━━━━━━━")
        for signal in sell_signals_sorted:
            lines.append(f"• *{signal['ticker']}* - ₹{signal['current_price']:.2f}")
            lines.append(f"  RSI: {signal['current_rsi']:.1f}")
        lines.append("")
    else:
        lines.append("🔴 *SELL SIGNALS*: None")
        lines.append("")
    
    # Footer
    if is_test:
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append("✅ Test completed successfully")
        lines.append("📱 Real subscribers were NOT notified")
    else:
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append("🤖 Automated Signal Bot")
        lines.append("🔄 Next scan at next scheduled time")
    
    return "\n".join(lines)


def split_message(text: str, max_length: int = 3800) -> List[str]:
    """
    Split a message into smaller chunks that fit within Telegram's limit.
    
    Args:
        text: The message text to split
        max_length: Maximum characters per message (default 3800 to leave room for formatting)
        
    Returns:
        List of message chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    lines = text.split('\n')
    
    for line in lines:
        line_length = len(line) + 1  # +1 for newline
        
        if current_length + line_length <= max_length:
            current_chunk.append(line)
            current_length += line_length
        else:
            # Save current chunk and start a new one
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Ensure we have at least one chunk
    if not chunks:
        chunks = [text]
    
    return chunks


def format_portfolio_message(user_name: str, portfolio: List[dict]) -> str:
    """Format user's portfolio into a Telegram message."""
    lines = []
    
    lines.append(f"📊 *{user_name}'s Portfolio*")
    lines.append("━━━━━━━━━━━━━━━━")
    lines.append("")
    
    if not portfolio:
        lines.append("Your portfolio is empty.")
        lines.append("")
        lines.append("Use /add <ticker> to add stocks to your portfolio")
        return "\n".join(lines)
    
    # Summary
    buy_count = sum(1 for p in portfolio if p["signal"] == "BUY")
    sell_count = sum(1 for p in portfolio if p["signal"] == "SELL")
    hold_count = sum(1 for p in portfolio if p["signal"] == "HOLD")
    
    lines.append(f"📈 *Portfolio Summary*")
    lines.append(f"🟢 Buy: {buy_count}")
    lines.append(f"🔴 Sell: {sell_count}")
    lines.append(f"🟡 Hold: {hold_count}")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━")
    lines.append("")
    
    # Sort alphabetically by ticker (ignoring signal type)
    portfolio_sorted = sorted(portfolio, key=lambda x: x['ticker'])
    
    # Holdings
    for item in portfolio_sorted:
        lines.append(f"{item['signal_emoji']} *{item['ticker']}* - *{item['signal']}*")
        lines.append(f"   💰 Price: ₹{item['current_price']:.2f}")
        lines.append(f"   📊 RSI: {item['current_rsi']:.1f}")
        lines.append(f"   📉 MACD: {item['macd_histogram']:+.4f}")
        lines.append("")
    
    lines.append("━━━━━━━━━━━━━━━━")
    lines.append("🤖 Stock Signal Bot")
    
    return "\n".join(lines)


def format_suggest_message(result: dict) -> str:
    """Format single ticker suggestion into a Telegram message."""
    lines = []
    
    # Convert signal to display format (WAIT -> Wait, others as-is)
    signal_display = result['signal']
    if signal_display == "WAIT":
        signal_display = "Wait"
    
    lines.append(f"📊 *{result['ticker']} Analysis*")
    lines.append("━━━━━━━━━━━━━━━━")
    lines.append("")
    
    lines.append(f"{result['signal_emoji']} *Signal: {signal_display}*")
    lines.append("")
    
    lines.append(f"💰 *Price:* ₹{result['current_price']:.2f}")
    lines.append("")
    
    lines.append("*Technical Indicators*")
    lines.append(f"📊 RSI (14): {result['current_rsi']:.1f}")
    lines.append(f"📉 MACD: {result['current_macd']:.4f}")
    lines.append(f"📈 Signal: {result['current_signal']:.4f}")
    lines.append(f"📊 Histogram: {result['macd_histogram']:+.4f}")
    lines.append("")
    
    lines.append("*RSI Levels*")
    lines.append("• Oversold: < 30")
    lines.append("• Neutral: 30-70")
    lines.append("• Overbought: > 70")
    lines.append("")
    
    if result["crossover_type"]:
        lines.append(f"⚡ Recent: {result['crossover_type'].title()} crossover detected")
    else:
        lines.append("⚡ Recent: No significant crossover")
    
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━")
    lines.append(f"🕐 Analyzed at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    return "\n".join(lines)


# ==================== TELEGRAM HANDLERS ====================

def load_subscribers() -> Dict[str, dict]:
    """Load subscribers from JSON file."""
    if subscribers_file.exists():
        try:
            with open(subscribers_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading subscribers: {e}")
            return {}
    return {}


def save_subscribers(subscribers: Dict[str, dict]) -> None:
    """Save subscribers to JSON file."""
    with open(subscribers_file, "w") as f:
        json.dump(subscribers, f, indent=2)


def is_admin(user_id: int) -> bool:
    """Check if user is an admin."""
    return user_id in ADMIN_IDS


async def send_notification(
    application: Application,
    message: str,
    recipient_ids: List[int] = None
) -> Tuple[int, int]:
    """Send notification to all subscribers or specific recipients."""
    subscribers = load_subscribers()
    
    if recipient_ids is None:
        recipient_ids = [
            user_id for user_id, data in subscribers.items()
            if data.get("active", True)
        ]
    
    success = 0
    failed = 0
    
    for user_id in recipient_ids:
        try:
            await application.bot.send_message(
                chat_id=int(user_id),
                text=message,
            )
            success += 1
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Failed to send to {user_id}: {e}")
            failed += 1
    
    logger.info(f"Notification sent: {success} success, {failed} failed")
    return success, failed


async def scheduled_analysis(application: Application) -> None:
    """Run scheduled analysis and send notifications to all subscribers."""
    logger.info("Starting scheduled analysis")
    
    # Run market-wide analysis
    buy_signals, sell_signals, summary = run_analysis()
    
    # Save signals for web dashboard (only market signals, no portfolio info)
    save_signals_for_web(buy_signals, sell_signals, summary)
    
    market_message = format_signal_message(buy_signals, sell_signals, summary, is_test=False)
    
    # Free memory from analysis results after formatting
    del buy_signals, sell_signals, summary
    
    # Get all active subscribers
    subscribers = load_subscribers()
    active_subscribers = [
        user_id for user_id, data in subscribers.items()
        if data.get("active", True)
    ]
    
    success = 0
    failed = 0
    
    for user_id in active_subscribers:
        try:
            # Get user's portfolio
            user_portfolio_tickers = get_user_portfolio(user_id)
            
            if user_portfolio_tickers:
                # Analyze user's portfolio (use_hold=True since these are portfolio holdings)
                portfolio_data = []
                for ticker in user_portfolio_tickers:
                    result = analyze_ticker_signal(ticker, use_hold=True)
                    if result:
                        portfolio_data.append(result)
                
                # Create combined message (market + portfolio)
                user_data = subscribers.get(user_id, {})
                user_name = user_data.get("name", "User")
                combined_message = format_scheduled_notification(
                    market_message, user_name, portfolio_data
                )
            else:
                # No portfolio - just send market message
                combined_message = market_message
            
            # Split message if too long and send all chunks
            messages = split_message(combined_message)
            
            for msg in messages:
                await application.bot.send_message(
                    chat_id=int(user_id),
                    text=msg,
                )
            success += 1
            
            # Cleanup portfolio data to free memory
            del portfolio_data, combined_message, messages
            
            # Rate limiting - increased to reduce CPU usage
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to send notification to {user_id}: {e}")
            failed += 1
    
    # Force garbage collection after analysis to prevent memory buildup
    gc.collect()
    
    # Cleanup remaining large variables
    del market_message, subscribers, active_subscribers
    
    logger.info(f"Scheduled notification sent: {success} delivered, {failed} failed")


def format_scheduled_notification(market_message: str, user_name: str, portfolio: List[dict]) -> str:
    """Format combined market and portfolio notification."""
    lines = []
    
    lines.append(f"📊 *Daily Market & Portfolio Update*")
    lines.append("━━━━━━━━━━━━━━━━")
    lines.append(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Portfolio Summary
    if portfolio:
        buy_count = sum(1 for p in portfolio if p["signal"] == "BUY")
        sell_count = sum(1 for p in portfolio if p["signal"] == "SELL")
        hold_count = sum(1 for p in portfolio if p["signal"] == "WAIT")
        
        lines.append(f"💼 *{user_name}'s Portfolio*")
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append(f"🟢 Buy: {buy_count} | 🟡 Hold: {hold_count} | 🔴 Sell: {sell_count}")
        lines.append("")
        
        # Sort alphabetically by ticker
        portfolio_sorted = sorted(portfolio, key=lambda x: x['ticker'])
        
        # Show portfolio holdings
        for item in portfolio_sorted:
            lines.append(f"{item['signal_emoji']} *{item['ticker']}* - {item['signal']} @ ₹{item['current_price']:.2f}")
        
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append("")
    
    # Add market scan (remove the footer first)
    market_lines = market_message.split("\n")
    # Remove the last 3 lines (footer)
    market_lines = market_lines[:-3]
    lines.extend(market_lines)
    
    return "\n".join(lines)


async def run_test(application: Application, user_id: int) -> str:
    """Run a test analysis and send results only to the admin."""
    global test_mode
    test_mode = True
    
    try:
        await application.bot.send_message(
            chat_id=user_id,
            text="🧪 *Test Mode Activated*\n\n⏳ Analyzing tickers...",
        )
        
        buy_signals, sell_signals, summary = run_analysis()
        message = format_signal_message(buy_signals, sell_signals, summary, is_test=True)
        
        # Cleanup analysis data
        del buy_signals, sell_signals, summary
        
        # Split message if too long and send all chunks
        messages = split_message(message)
        
        for msg in messages:
            await application.bot.send_message(
                chat_id=user_id,
                text=msg,
            )
        
        # Cleanup remaining data
        del message, messages
        
        return f"✅ Test completed. Signals sent to you only."
        
    except Exception as e:
        error_msg = f"❌ Test failed: {str(e)}"
        logger.error(f"Test error: {e}")
        return error_msg
    finally:
        test_mode = False


# Command Handlers

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command - Subscribe user to notifications."""
    user = update.effective_user
    user_id = str(user.id)
    
    subscribers = load_subscribers()
    
    if user_id in subscribers:
        message = (
            f"👋 Welcome back, *{user.first_name}*!\n\n"
            "You are already subscribed to signal notifications.\n\n"
            "📊 *New Features:*\n"
            "• /add <ticker> - Add stock to your portfolio\n"
            "• /remove <ticker> - Remove stock from portfolio\n"
            "• /portfolio - View your portfolio with signals\n"
            "• /suggest <ticker> - Get analysis for any stock\n\n"
            "Use /help for all commands."
        )
    else:
        subscribers[user_id] = {
            "name": f"{user.first_name} {user.last_name or ''}".strip(),
            "username": user.username,
            "active": True,
            "subscribed_at": datetime.now().isoformat(),
        }
        save_subscribers(subscribers)
        
        message = (
            f"👋 Hello, *{user.first_name}*!\n\n"
            "✅ You have been subscribed to stock signal notifications.\n\n"
            "📊 *Available Commands:*\n"
            "• /signals - Get current buy/sell signals\n"
            "• /add <ticker> - Add stock to your portfolio\n"
            "• /remove <ticker> - Remove stock from portfolio\n"
            "• /portfolio - View your portfolio with signals\n"
            "• /suggest <ticker> - Get analysis for any stock\n\n"
            "Use /help to see all commands."
        )
    
    await update.message.reply_html(message)


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop command - Unsubscribe user from notifications."""
    user = update.effective_user
    user_id = str(user.id)
    
    subscribers = load_subscribers()
    
    if user_id in subscribers:
        subscribers[user_id]["active"] = False
        subscribers[user_id]["unsubscribed_at"] = datetime.now().isoformat()
        save_subscribers(subscribers)
        
        message = (
            f"👋 Goodbye, *{user.first_name}*!\n\n"
            "❌ You have been unsubscribed from signal notifications.\n"
            "Use /start to subscribe again."
        )
    else:
        message = (
            "❓ You are not currently subscribed.\n"
            "Use /start to subscribe."
        )
    
    await update.message.reply_html(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command - Show available commands."""
    message = """
📚 *Available Commands*

🟢 *Subscription*
• /start - Subscribe to signal notifications
• /stop - Unsubscribe from notifications

📊 *Signals*
• /signals - Get current buy/sell signals
• /suggest <ticker> - Get analysis for any stock (Buy/Sell/Wait)
• /depth <ticker> - Get LLM-powered deep analysis with comprehensive report

💼 *Portfolio Management*
• /add <ticker> - Add stock to your portfolio
• /remove <ticker> - Remove stock from portfolio
• /portfolio - View your portfolio with signals

🧪 *Admin*
• /test - Run test analysis (Admin only)
• /upload - Get upload instructions (Admin only)
• /status - Check bot status

📈 *Signal Types*
• 🟢 BUY - MACD bullish crossover + RSI OK
• 🟡 WAIT - No clear signal (for /suggest)
• 🟡 HOLD - No clear signal (for /portfolio)
• 🔴 SELL - MACD bearish crossover

🤖 *About*
• Automated Stock Signal Scanner
• No financial advice provided
    """
    
    await update.message.reply_markdown(message)


async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /signals command - Get current signals on demand."""
    await update.message.reply_text("📊 Analyzing current market data...")
    
    buy_signals, sell_signals, summary = run_analysis()
    message = format_signal_message(buy_signals, sell_signals, summary, is_test=False)
    
    # Cleanup analysis data
    del buy_signals, sell_signals, summary
    
    # Split message if too long
    messages = split_message(message)
    
    for i, msg in enumerate(messages):
        if i == 0:
            await update.message.reply_markdown(msg)
        else:
            await update.message.reply_markdown(msg)
    
    # Cleanup remaining data
    del message, messages


async def add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /add command - Add ticker to user's portfolio."""
    user = update.effective_user
    user_id = str(user.id)
    
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a ticker symbol.\n"
            "Usage: /add <ticker>\n"
            "Example: /add AAPL"
        )
        return
    
    ticker = context.args[0].upper().strip()
    
    # Validate ticker exists
    await update.message.reply_text(f"🔍 Checking {ticker}...")
    
    result = analyze_ticker_signal(ticker)
    if result is None:
        await update.message.reply_text(
            f"❌ Ticker '{ticker}' not found or has insufficient data.\n"
            "Please check the symbol and try again."
        )
        return
    
    # Add to portfolio
    success, msg = add_to_portfolio(user_id, user.username or "", ticker)
    await update.message.reply_text(msg)


async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /remove command - Remove ticker from user's portfolio."""
    user_id = str(update.effective_user.id)
    
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a ticker symbol.\n"
            "Usage: /remove <ticker>\n"
            "Example: /remove AAPL"
        )
        return
    
    ticker = context.args[0].upper().strip()
    success, msg = remove_from_portfolio(user_id, ticker)
    await update.message.reply_text(msg)


async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /portfolio command - Show user's portfolio with signals."""
    user = update.effective_user
    user_id = str(user.id)
    
    await update.message.reply_text("📊 Analyzing your portfolio...")
    
    # Get user's portfolio tickers
    tickers = get_user_portfolio(user_id)
    
    if not tickers:
        message = (
            f"📊 *{user.first_name}'s Portfolio*\n\n"
            "Your portfolio is empty.\n\n"
            "Use /add <ticker> to add stocks to your portfolio.\n"
            "Example: /add AAPL"
        )
        await update.message.reply_markdown(message)
        return
    
    # Analyze each ticker (use_hold=True since these are portfolio holdings)
    portfolio_data = []
    errors = []
    
    for ticker in tickers:
        result = analyze_ticker_signal(ticker, use_hold=True)
        if result:
            portfolio_data.append(result)
        else:
            errors.append(ticker)
    
    # Format and send message
    message = format_portfolio_message(user.first_name, portfolio_data)
    
    if errors:
        message += f"\n⚠️ Could not analyze: {', '.join(errors)}"
    
    # Cleanup portfolio data
    del portfolio_data, tickers
    
    # Split message if too long
    messages = split_message(message)
    
    for msg in messages:
        await update.message.reply_markdown(msg)
    
    # Cleanup remaining data
    del message, messages, errors


async def suggest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /suggest command - Get analysis for any ticker."""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a ticker symbol.\n"
            "Usage: /suggest <ticker>\n"
            "Example: /suggest AAPL"
        )
        return
    
    user_id = str(update.effective_user.id)
    ticker = context.args[0].upper().strip()
    
    # Check if ticker is in user's portfolio
    user_portfolio = get_user_portfolio(user_id)
    in_portfolio = ticker.upper() in [t.upper() for t in user_portfolio]
    
    await update.message.reply_text(f"🔍 Analyzing {ticker}...")
    
    # Use HOLD if in portfolio, WAIT if not in portfolio
    result = analyze_ticker_signal(ticker, use_hold=in_portfolio)
    
    if result is None:
        await update.message.reply_text(
            f"❌ Ticker '{ticker}' not found or has insufficient data.\n"
            "Please check the symbol and try again."
        )
        return
    
    message = format_suggest_message(result)
    await update.message.reply_markdown(message)
    
    # Cleanup
    del result, message, user_portfolio


def cleanup_symbol_folder(symbol: str) -> bool:
    """Clean up the {symbol}_data folder and related files after analysis."""
    symbol = symbol.upper().strip()
    base_path = Path(__file__).parent
    folder_path = base_path / f"{symbol}_data"
    db_path = base_path / f"{symbol}_stock_data.db"
    rec_path = base_path / f"{symbol}_recommendation.txt"
    
    try:
        # Remove the data folder if it exists
        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)
            logger.info(f"Removed data folder: {folder_path}")
        
        # Remove any stray database files
        if db_path.exists():
            os.remove(db_path)
            logger.info(f"Removed database file: {db_path}")
        
        # Remove recommendation files
        if rec_path.exists():
            os.remove(rec_path)
            logger.info(f"Removed recommendation file: {rec_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error cleaning up files for {symbol}: {e}")
        return False


async def depth_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /depth command - Get LLM-powered deep analysis for any ticker."""
    # Double-check that LLMStockRecommender is available
    if not LLM_AVAILABLE or 'LLMStockRecommender' not in globals():
        error_msg = "❌ LLM depth analysis is not available.\n\n"
        error_msg += "The llm_stock_recommendation.py module could not be imported properly.\n"
        if LLM_IMPORT_ERROR:
            error_msg += f"\nError details: {LLM_IMPORT_ERROR}\n"
        error_msg += "\nPlease check that all files are in the same directory:\n"
        error_msg += "• bot.py\n• llm_stock_recommendation.py\n• nse_india_api.py\n• signal_generator.py"
        await update.message.reply_text(error_msg)
        logger.error(f"LLM not available. Import error: {LLM_IMPORT_ERROR}")
        return
    
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a ticker symbol.\n"
            "Usage: /depth <ticker>\n"
            "Example: /depth AAPL\n\n"
            "This command provides a comprehensive LLM-powered analysis including:\n"
            "• Technical indicators (RSI, MACD, Moving Averages)\n"
            "• Smart money indicators\n"
            "• Options market data (PCR, Max Pain)\n"
            "• FII/DII institutional flows\n"
            "• Sector analysis and relative strength"
        )
        return
    
    ticker = context.args[0].upper().strip()
    
    await update.message.reply_text(
        f"🔍 *Deep Analysis for {ticker}*\n\n"
        f"⏳ Fetching comprehensive stock data and running LLM analysis...\n"
        f"This may take 1-2 minutes. Please wait...",
        parse_mode='Markdown'
    )
    
    try:
        # Double-check LLMStockRecommender is defined
        if 'LLMStockRecommender' not in globals():
            raise NameError("LLMStockRecommender is not defined in global scope")
        
        # Create recommender and run analysis
        recommender = LLMStockRecommender()
        
        # Run in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(recommender.run, symbol=ticker)
        
        if result is None or 'full_response' not in result:
            await update.message.reply_text(
                f"❌ Analysis failed for {ticker}.\n"
                "Please check the symbol and try again."
            )
            cleanup_symbol_folder(ticker)
            return
        
        # Get the full response
        response = result.get('full_response', '')
        
        if not response:
            await update.message.reply_text(
                f"❌ No analysis result received for {ticker}."
            )
            cleanup_symbol_folder(ticker)
            return
        
        # Format and send the response
        formatted_message = format_depth_message(ticker, response)
        
        # Send in chunks if too long (Telegram limit is 4096 chars per message)
        chunks = split_message(formatted_message, max_length=3800)
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                await update.message.reply_markdown(chunk)
            else:
                await update.message.reply_markdown(chunk)
        
        logger.info(f"Depth analysis completed for {ticker}")
        
    except Exception as e:
        logger.error(f"Error in depth analysis for {ticker}: {e}", exc_info=True)
        error_msg = f"❌ An error occurred while analyzing {ticker}:\n\n{str(e)}\n\n"
        error_msg += "Please check the bot logs for more details."
        await update.message.reply_text(error_msg)
    finally:
        # Always cleanup, regardless of success or failure
        cleanup_symbol_folder(ticker)


def format_depth_message(ticker: str, response: str) -> str:
    """Format the LLM response for Telegram display."""
    # Add a header to the response
    lines = []
    lines.append(f"📊 *Deep Analysis: {ticker}* (LLM Powered)")
    lines.append("━" * 30)
    lines.append("")
    
    # Add the LLM response
    lines.append(response)
    
    lines.append("")
    lines.append("━" * 30)
    lines.append(f"🕐 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)


async def upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /upload command - Instructions for uploading ticker file."""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text(
            "❌ This command is only available to administrators."
        )
        return
    
    message = (
        "📤 UPLOAD TICKER FILE\n\n"
        "To update the ticker list, simply send me the Excel file directly.\n\n"
        "REQUIREMENTS:\n"
        "• File format: Excel (.xlsx or .xls)\n"
        "• First column must contain ticker symbols\n"
        "• One ticker per row\n\n"
        "TIP: Just drag and drop the file or attach it to your message.\n\n"
        "NOTE: The old ticker.xlsx will be backed up as ticker_backup.xlsx"
    )
    
    await update.message.reply_text(message)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads - Admin only to update ticker.xlsx."""
    user = update.effective_user
    user_id = user.id
    
    # Check if admin
    if not is_admin(user_id):
        await update.message.reply_text(
            "❌ You are not authorized to upload files."
        )
        return
    
    # Get the document
    document = update.message.document
    if not document:
        await update.message.reply_text(
            "❌ No document found."
        )
        return
    
    # Check file type
    file_name = document.file_name.lower()
    if not (file_name.endswith('.xlsx') or file_name.endswith('.xls')):
        await update.message.reply_text(
            "Invalid file format. Please upload an Excel file (.xlsx or .xls)"
        )
        return
    
    # Check file size (max 10MB)
    if document.file_size > 10 * 1024 * 1024:
        await update.message.reply_text(
            "❌ File too large.\n"
            "Maximum file size is 10MB"
        )
        return
    
    await update.message.reply_text(
        f"📥 Downloading {document.file_name}..."
    )
    
    try:
        # Get the file
        file = await context.bot.get_file(document.file_id)
        
        # Create backup of old file if exists
        if tickers_file.exists():
            backup_file = Path(__file__).parent / "tickers_backup.xlsx"
            tickers_file.replace(backup_file)
            logger.info(f"Backed up old ticker file to {backup_file}")
        
        # Download new file
        new_file_path = tickers_file
        await file.download_to_drive(new_file_path)
        
        # Validate the file
        try:
            df = pd.read_excel(new_file_path)
            tickers = df.iloc[:, 0].dropna().astype(str).unique().tolist()
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            
            await update.message.reply_text(
                f"FILE UPLOADED SUCCESSFULLY!\n\n"
                f"DETAILS:\n"
                f"- File: {document.file_name}\n"
                f"- Tickers: {len(tickers)}\n"
                f"- Backup: tickers_backup.xlsx\n\n"
                f"NOTE: Users may need to run /signals to get updated signals."
            )
            
            logger.info(f"Ticker file updated: {len(tickers)} tickers loaded")
            
        except Exception as e:
            # Restore backup if validation fails
            if backup_file.exists() and not tickers_file.exists():
                backup_file.replace(tickers_file)
            
            await update.message.reply_text(
                f"ERROR validating file:\n\n{str(e)}\n\n"
                f"The original file has been restored."
            )
            logger.error(f"Error validating uploaded file: {e}")
    
    except Exception as e:
        await update.message.reply_text(
            f"ERROR downloading file:\n\n{str(e)}"
        )
        logger.error(f"Error downloading file: {e}")


async def test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /test command - Admin-only test mode."""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text(
            "❌ This command is only available to administrators."
        )
        return
    
    global _bot_application
    result = await run_test(_bot_application, user_id)
    await update.message.reply_text(result)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - Check subscription and system status."""
    user_id = str(update.effective_user.id)
    
    subscribers = load_subscribers()
    user_tickers = get_user_portfolio(user_id)
    subscriber_count = len([s for s in subscribers.values() if s.get("active", True)])
    
    message = (
        f"📊 *System Status*\n\n"
        f"🗂️ Tickers in database: {len(load_tickers())}\n"
        f"👥 Active subscribers: {subscriber_count}\n"
        f"💼 Your portfolio: {len(user_tickers)} stocks\n\n"
    )
    
    if user_id in subscribers:
        user_data = subscribers[user_id]
        is_active = user_data.get("active", False)
        status_text = "✅ Active" if is_active else "❌ Unsubscribed"
        
        message += (
            f"*Your Status*\n"
            f"• Subscription: {status_text}\n"
            f"• Subscribed at: {user_data.get('subscribed_at', 'Unknown')}"
        )
    else:
        message += "*Your Status*\n• Not subscribed. Use /start to subscribe."
    
    await update.message.reply_markdown(message)


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    await update.message.reply_text(
        "❓ Unknown command. Use /help to see available commands."
    )


# ==================== SCHEDULER ====================

def _run_scheduled_analysis_sync():
    """Helper function to run scheduled analysis from sync scheduler."""
    global _bot_application
    if _bot_application is not None:
        try:
            asyncio.run(_run_scheduled_analysis(_bot_application))
        except Exception as e:
            logger.error(f"Error in scheduled analysis: {e}")


async def _run_scheduled_analysis(application: Application) -> None:
    """Helper function to run scheduled analysis."""
    await scheduled_analysis(application)


def setup_scheduler(application: Application) -> BackgroundScheduler:
    """Setup the APScheduler for automated analysis."""
    global _bot_application
    _bot_application = application
    
    scheduler = BackgroundScheduler()
    
    for time_str in SCHEDULED_TIMES:
        hour, minute = map(int, time_str.split(":"))
        trigger = CronTrigger(hour=hour, minute=minute)
        
        scheduler.add_job(
            _run_scheduled_analysis_sync,
            trigger=trigger,
            id=f"analysis_{time_str}",
            name=f"Analysis at {time_str}",
            replace_existing=True,
        )
        
        logger.info(f"Scheduled analysis at {time_str}")
    
    scheduler.start()
    logger.info("Scheduler started")
    
    return scheduler


# ==================== WEB DASHBOARD ROUTES ====================

# Global variables for depth analysis
from concurrent.futures import ThreadPoolExecutor
import threading

_web_executor = ThreadPoolExecutor(max_workers=2)
_web_active_analysis = {}
_web_analysis_lock = threading.Lock()


def _create_flask_app() -> Flask:
    """Create and configure the Flask web application."""
    app = Flask(__name__, 
        template_folder='templates',
        static_folder='static'
    )
    CORS(app)
    
    @app.route('/')
    def index():
        """Render the main dashboard page."""
        return render_template('index.html', 
                              llm_available=LLM_AVAILABLE,
                              web_integration=True)
    
    @app.route('/api/signals')
    def api_signals():
        """API endpoint to get current signals."""
        signals_data = get_current_signals()
        return jsonify(signals_data)
    
    @app.route('/api/signals/history')
    def api_signals_history():
        """API endpoint to get signal history."""
        limit = request.args.get('limit', 20, type=int)
        history = get_signals_history(limit=limit)
        return jsonify({"history": history})
    
    @app.route('/api/signals/html')
    def api_signals_html():
        """API endpoint to get signals as rendered HTML."""
        signals_data = get_current_signals()
        buy_signals = signals_data.get('buy_signals', [])
        sell_signals = signals_data.get('sell_signals', [])
        html = format_signals_for_web(buy_signals, sell_signals)
        return Response(
            html,
            mimetype='text/html',
            headers={'Cache-Control': 'no-cache'}
        )
    
    @app.route('/api/depth', methods=['POST'])
    def api_depth_analysis():
        """API endpoint for depth analysis."""
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                "error": "Missing required parameter: symbol",
                "status": "error"
            }), 400
        
        symbol = data.get('symbol', '').strip().upper()
        
        if not symbol:
            return jsonify({
                "error": "Invalid symbol provided",
                "status": "error"
            }), 400
        
        # Check if LLM is available
        if not LLM_AVAILABLE:
            error_msg = "LLM depth analysis is not available."
            return jsonify({
                "error": error_msg,
                "status": "error",
                "symbol": symbol
            }), 503
        
        # Check if analysis is already running for this symbol
        task_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        with _web_analysis_lock:
            if symbol in _web_active_analysis:
                future = _web_active_analysis[symbol]
                if not future.done():
                    return jsonify({
                        "message": f"Analysis for {symbol} is already in progress",
                        "status": "processing",
                        "symbol": symbol
                    })
            
            # Start new analysis task
            future = _web_executor.submit(_run_depth_analysis, symbol)
            _web_active_analysis[symbol] = future
            task_id = symbol
        
        return jsonify({
            "message": f"Analysis started for {symbol}",
            "status": "processing",
            "symbol": symbol,
            "task_id": task_id
        })
    
    @app.route('/api/depth/result/<task_id>')
    def api_depth_result(task_id):
        """API endpoint to get depth analysis result."""
        symbol = task_id.split('_')[0] if '_' in task_id else task_id
        
        with _web_analysis_lock:
            if symbol in _web_active_analysis:
                future = _web_active_analysis[symbol]
                
                if future.done():
                    try:
                        result = future.result(timeout=5)
                        del _web_active_analysis[symbol]
                        return jsonify(result)
                    except Exception as e:
                        del _web_active_analysis[symbol]
                        return jsonify({
                            "error": str(e),
                            "status": "error",
                            "symbol": symbol
                        }), 500
                else:
                    return jsonify({
                        "status": "processing",
                        "symbol": symbol,
                        "message": "Analysis in progress... please wait"
                    })
            else:
                return jsonify({
                    "error": "Task not found",
                    "status": "error",
                    "task_id": task_id
                }), 404
    
    @app.route('/api/depth/sync', methods=['POST'])
    def api_depth_analysis_sync():
        """Synchronous depth analysis endpoint."""
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                "error": "Missing required parameter: symbol",
                "status": "error"
            }), 400
        
        symbol = data.get('symbol', '').strip().upper()
        
        if not symbol:
            return jsonify({
                "error": "Invalid symbol provided",
                "status": "error"
            }), 400
        
        if not LLM_AVAILABLE:
            error_msg = "LLM depth analysis is not available."
            return jsonify({
                "error": error_msg,
                "status": "error",
                "symbol": symbol
            }), 503
        
        try:
            result = _run_depth_analysis(symbol)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in depth analysis: {e}")
            return jsonify({
                "error": str(e),
                "status": "error",
                "symbol": symbol
            }), 500
    
    @app.route('/api/status')
    def api_status():
        """API endpoint to get system status."""
        return jsonify({
            "status": "online",
            "llm_available": LLM_AVAILABLE,
            "web_integration": True,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        })
    
    @app.route('/health')
    def health_check():
        """Health check endpoint for monitoring."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    
    return app


def _run_depth_analysis(symbol: str) -> dict:
    """Run depth analysis for a symbol."""
    try:
        if 'LLMStockRecommender' not in globals():
            return {
                "status": "error",
                "symbol": symbol,
                "error": "LLMStockRecommender not available"
            }
        
        recommender = LLMStockRecommender()
        result = recommender.run(symbol=symbol)
        
        if result and 'full_response' in result:
            return {
                "status": "success",
                "symbol": symbol,
                "recommendation": result.get('recommendation', 'NEUTRAL'),
                "analysis": result.get('full_response', ''),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "symbol": symbol,
                "error": "Analysis failed to produce results",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Depth analysis error for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _run_flask_server():
    """Run the Flask web server in a blocking manner."""
    global _flask_app
    
    if not FLASK_AVAILABLE:
        logger.warning("Flask not available. Web dashboard will not start.")
        return
    
    try:
        _flask_app = _create_flask_app()
        logger.info("Starting web dashboard server on port 5000...")
        _flask_app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Error starting web server: {e}")


def start_web_dashboard() -> threading.Thread:
    """Start the web dashboard in a background thread."""
    if not FLASK_AVAILABLE:
        logger.warning("Flask not installed. Web dashboard cannot be started.")
        return None
    
    web_thread = threading.Thread(
        target=_run_flask_server,
        name="FlaskWebServer",
        daemon=True
    )
    web_thread.start()
    logger.info("Web dashboard thread started")
    return web_thread


# ==================== MAIN ====================

def main() -> None:
    """Main entry point."""
    print("="*60)
    print("Stock Signal Bot with Web Dashboard")
    print("="*60)
    print()
    
    logger.info("Starting Stock Signal Bot")
    
    # Initialize database
    init_database()
    
    # Create application
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(1)
        .build()
    )
    
    global _bot_application
    _bot_application = application
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("signals", signals))
    application.add_handler(CommandHandler("add", add))
    application.add_handler(CommandHandler("remove", remove))
    application.add_handler(CommandHandler("portfolio", portfolio))
    application.add_handler(CommandHandler("suggest", suggest))
    application.add_handler(CommandHandler("depth", depth_command))
    application.add_handler(CommandHandler("upload", upload))
    application.add_handler(CommandHandler("test", test))
    application.add_handler(CommandHandler("status", status))
    
    # Handle document uploads (for ticker file updates)
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    # Handle unknown commands
    application.add_handler(MessageHandler(filters.COMMAND, unknown))
    
    # Setup scheduler
    scheduler = setup_scheduler(application)
    
    # Print status
    print(f"Telegram Bot: ✓ Running")
    print(f"LLM Depth Analysis: {'✓ Available' if LLM_AVAILABLE else '✗ Not Available'}")
    print(f"Web Dashboard: {'✓ Enabled (Port 5000)' if FLASK_AVAILABLE else '✗ Disabled (Flask not installed)'}")
    print()
    
    # Start web dashboard in background thread
    if FLASK_AVAILABLE:
        web_thread = start_web_dashboard()
        print(f"Web Dashboard: http://localhost:5000")
        print()
    else:
        print("Web Dashboard: ⚠️ Flask not installed. Install with: pip install flask flask-cors")
        print()
    
    # Run the bot
    print("Bot is ready. Press Ctrl+C to stop.")
    print("="*60)
    
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        scheduler.shutdown()
        _web_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
