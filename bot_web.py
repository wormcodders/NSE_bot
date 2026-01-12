"""
Stock Signal Bot - Web Application Integration Module

This module provides web interface functionality for the stock signal bot.
It saves scheduled analysis results to JSON files for the web dashboard to display.

Usage:
    Import this module in bot.py to enable web dashboard features:
    from bot_web import save_signals_for_web, get_current_signals
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Web data directory
WEB_DATA_DIR = Path(__file__).parent / "web_data"
WEB_DATA_DIR.mkdir(exist_ok=True)

# Signals history file
SIGNALS_FILE = WEB_DATA_DIR / "signals_history.json"

# Current signals file (for real-time display)
CURRENT_SIGNALS_FILE = WEB_DATA_DIR / "current_signals.json"


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
        update_signals_history(signal_data)
        
        logger.info(f"Signals saved for web: {len(buy_signals)} buy, {len(sell_signals)} sell")
        return True
        
    except Exception as e:
        logger.error(f"Error saving signals for web: {e}")
        return False


def update_signals_history(new_signal_data: dict) -> None:
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
                    <div class="signal-price">₹{signal.get('current_price', 0):.2f}</div>
                    <div class="signal-indicators">
                        <span class="indicator">RSI: {signal.get('current_rsi', 0):.1f}</span>
                        <span class="indicator">MACD: {signal.get('macd_histogram', 0):+.4f}</span>
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
                    <div class="signal-price">₹{signal.get('current_price', 0):.2f}</div>
                    <div class="signal-indicators">
                        <span class="indicator">RSI: {signal.get('current_rsi', 0):.1f}</span>
                        <span class="indicator">MACD: {signal.get('macd_histogram', 0):+.4f}</span>
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
