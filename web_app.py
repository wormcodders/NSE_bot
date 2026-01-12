#!/usr/bin/env python3
"""
Stock Signal Bot - Web Application

Flask web application for displaying stock signals and depth analysis.
This provides a web interface that mirrors the Telegram bot functionality.

Features:
1. Live Signals Dashboard - Shows scheduled analysis results
2. Depth Analysis - Search for comprehensive stock analysis

Author: MiniMax Agent
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

# Import bot modules
try:
    from bot_web import (
        get_current_signals, 
        get_signals_history,
        format_signals_for_web
    )
    WEB_INTEGRATION_AVAILABLE = True
except ImportError:
    WEB_INTEGRATION_AVAILABLE = False
    print("Warning: bot_web module not found. Signals display will show placeholder data.")

# Import LLM recommender for depth analysis
LLM_AVAILABLE = False
LLMRecommenderClass = None
LLM_IMPORT_ERROR = None

try:
    import llm_stock_recommendation
    if hasattr(llm_stock_recommendation, 'LLMStockRecommender'):
        LLMRecommenderClass = llm_stock_recommendation.LLMStockRecommender
        LLM_AVAILABLE = True
        print("✓ LLMStockRecommender available for depth analysis")
    else:
        LLM_IMPORT_ERROR = "LLMStockRecommender class not found"
except ImportError as e:
    LLM_IMPORT_ERROR = str(e)
    print(f"✗ LLM import error: {e}")
except Exception as e:
    LLM_IMPORT_ERROR = str(e)
    print(f"✗ Unexpected LLM import error: {e}")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)

# Thread pool for depth analysis (runs in background)
executor = ThreadPoolExecutor(max_workers=2)

# Store active analysis tasks
active_analysis = {}
analysis_lock = threading.Lock()


@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html', 
                          llm_available=LLM_AVAILABLE,
                          web_integration=WEB_INTEGRATION_AVAILABLE)


@app.route('/api/signals')
def api_signals():
    """
    API endpoint to get current signals.
    Returns JSON data for the signals dashboard.
    """
    if WEB_INTEGRATION_AVAILABLE:
        signals_data = get_current_signals()
    else:
        # Return placeholder data if bot_web is not available
        signals_data = {
            "timestamp": None,
            "scheduled_time": None,
            "market_scan": {
                "total_tickers": 0,
                "analyzed": 0,
                "errors": 0,
                "execution_time": 0
            },
            "buy_signals": [],
            "sell_signals": [],
            "placeholder": True,
            "message": "Waiting for scheduled analysis..."
        }
    
    return jsonify(signals_data)


@app.route('/api/signals/history')
def api_signals_history():
    """
    API endpoint to get signal history.
    Returns list of historical signal entries.
    """
    limit = request.args.get('limit', 20, type=int)
    
    if WEB_INTEGRATION_AVAILABLE:
        history = get_signals_history(limit=limit)
    else:
        history = []
    
    return jsonify({"history": history})


@app.route('/api/signals/html')
def api_signals_html():
    """
    API endpoint to get signals as rendered HTML.
    Useful for iframe or partial updates.
    """
    if WEB_INTEGRATION_AVAILABLE:
        signals_data = get_current_signals()
        buy_signals = signals_data.get('buy_signals', [])
        sell_signals = signals_data.get('sell_signals', [])
        
        html = format_signals_for_web(buy_signals, sell_signals)
        
        return Response(
            html,
            mimetype='text/html',
            headers={'Cache-Control': 'no-cache'}
        )
    else:
        return Response(
            '<div class="no-signals"><p>Signals data not available. Run the bot to generate signals.</p></div>',
            mimetype='text/html'
        )


@app.route('/api/depth', methods=['POST'])
def api_depth_analysis():
    """
    API endpoint for depth analysis.
    Accepts POST with JSON: {"symbol": "TICKER"}
    Returns comprehensive analysis in JSON format.
    """
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
        if LLM_IMPORT_ERROR:
            error_msg += f" Error: {LLM_IMPORT_ERROR}"
        
        return jsonify({
            "error": error_msg,
            "status": "error",
            "symbol": symbol
        }), 503
    
    # Check if analysis is already running for this symbol
    task_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    with analysis_lock:
        if symbol in active_analysis:
            # Check if task is still running
            future = active_analysis[symbol]
            if not future.done():
                return jsonify({
                    "message": f"Analysis for {symbol} is already in progress",
                    "status": "processing",
                    "symbol": symbol
                })
        
        # Start new analysis task
        future = executor.submit(run_depth_analysis, symbol)
        active_analysis[symbol] = future
        task_id = symbol
    
    return jsonify({
        "message": f"Analysis started for {symbol}",
        "status": "processing",
        "symbol": symbol,
        "task_id": task_id
    })


@app.route('/api/depth/result/<task_id>')
def api_depth_result(task_id):
    """
    API endpoint to get depth analysis result.
    Used for polling status of async analysis.
    """
    # Extract symbol from task_id
    symbol = task_id.split('_')[0] if '_' in task_id else task_id
    
    with analysis_lock:
        if symbol in active_analysis:
            future = active_analysis[symbol]
            
            if future.done():
                try:
                    result = future.result(timeout=5)
                    # Remove from active tasks
                    del active_analysis[symbol]
                    return jsonify(result)
                except Exception as e:
                    del active_analysis[symbol]
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
    """
    Synchronous depth analysis endpoint.
    Waits for analysis to complete before returning.
    Use this for quick requests or when polling is not feasible.
    """
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
        if LLM_IMPORT_ERROR:
            error_msg += f" Error: {LLM_IMPORT_ERROR}"
        
        return jsonify({
            "error": error_msg,
            "status": "error",
            "symbol": symbol
        }), 503
    
    try:
        result = run_depth_analysis(symbol)
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
    """
    API endpoint to get system status.
    """
    return jsonify({
        "status": "online",
        "llm_available": LLM_AVAILABLE,
        "web_integration": WEB_INTEGRATION_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


def run_depth_analysis(symbol: str) -> dict:
    """
    Run depth analysis for a symbol.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Create recommender instance
        recommender = LLMRecommenderClass()
        
        # Run analysis
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
        raise e


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


def create_sample_signals():
    """Create sample signals data for demonstration if no real data exists."""
    web_data_dir = SCRIPT_DIR / "web_data"
    web_data_dir.mkdir(exist_ok=True)
    
    current_signals_file = web_data_dir / "current_signals.json"
    
    if not current_signals_file.exists():
        sample_signals = {
            "timestamp": datetime.now().isoformat(),
            "scheduled_time": "09:30",
            "market_scan": {
                "total_tickers": 50,
                "analyzed": 48,
                "errors": 2,
                "execution_time": 45.2
            },
            "buy_signals": [
                {
                    "ticker": "INFY",
                    "signal": "BUY",
                    "price": 1450.25,
                    "rsi": 42.5,
                    "macd": 2.34,
                    "signal_line": 1.89,
                    "histogram": 0.45,
                    "crossover_type": "bullish"
                },
                {
                    "ticker": "TCS",
                    "signal": "BUY",
                    "price": 3850.75,
                    "rsi": 38.2,
                    "macd": 5.67,
                    "signal_line": 4.89,
                    "histogram": 0.78,
                    "crossover_type": "bullish"
                }
            ],
            "sell_signals": [
                {
                    "ticker": "HDFCBANK",
                    "signal": "SELL",
                    "price": 1620.50,
                    "rsi": 68.5,
                    "macd": -1.23,
                    "signal_line": -0.89,
                    "histogram": -0.34,
                    "crossover_type": "bearish"
                }
            ],
            "sample_data": True
        }
        
        with open(current_signals_file, 'w', encoding='utf-8') as f:
            json.dump(sample_signals, f, indent=2, ensure_ascii=False)
        
        logger.info("Created sample signals data for demonstration")


if __name__ == "__main__":
    print("="*60)
    print("Stock Signal Bot - Web Dashboard")
    print("="*60)
    print()
    
    # Create sample data for demonstration
    create_sample_signals()
    
    # Print status
    print(f"LLM Depth Analysis: {'✓ Available' if LLM_AVAILABLE else '✗ Not Available'}")
    print(f"Web Integration: {'✓ Available' if WEB_INTEGRATION_AVAILABLE else '✗ Not Available'}")
    print()
    
    # Run the web server
    print("Starting web server...")
    print("Dashboard available at: http://localhost:5000")
    print("API endpoints:")
    print("  - GET  /api/signals       - Get current signals")
    print("  - GET  /api/signals/history - Get signal history")
    print("  - POST /api/depth         - Start depth analysis")
    print("  - POST /api/depth/sync    - Synchronous depth analysis")
    print()
    print("Press Ctrl+C to stop")
    print("="*60)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        executor.shutdown(wait=False)
