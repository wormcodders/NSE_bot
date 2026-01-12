"""
LLM Stock Recommendation System

This script integrates NSE India API data extraction with LLM-powered stock recommendations.
It fetches comprehensive stock data, generates trading signals, and provides AI-based
buy/sell/hold/wait recommendations.

Usage:
    python llm_stock_recommendation.py

The script will:
1. Prompt for a stock symbol
2. Extract all layers of stock data from NSE India
3. Generate compiled trading signals
4. Send data to LLM for analysis and recommendation
5. Display streaming recommendation with reasoning
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports (same directory as bot.py)
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from nse_india_api import NseIndia
from signal_generator import SignalGenerator


class LLMStockRecommender:
    """
    LLM-powered stock recommendation system that combines NSE India data
    with large language model analysis for investment decisions.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the stock recommender.
        
        Args:
            api_key: NVIDIA API key for accessing the LLM service
        """
        self.api_key = "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"
        self.client = None
        self.model = "moonshotai/kimi-k2-thinking"
        
    def initialize_llm_client(self):
        """Initialize the OpenAI-compatible LLM client for NVIDIA API."""
        if not self.api_key:
            print("Warning: No API key provided. LLM recommendations will not be available.")
            print("Please set the NVIDIA_API_KEY environment variable or provide it in the code.")
            return False
            
        try:
            from openai import OpenAI
            
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.api_key
            )
            return True
        except ImportError:
            print("Error: openai package not installed. Please install it with: pip install openai")
            return False
        except Exception as e:
            print(f"Error initializing LLM client: {e}")
            return False
    
    def get_stock_data(self, symbol: str) -> dict:
        """
        Extract comprehensive stock data for the given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'TCS', 'RELIANCE')
            
        Returns:
            Dictionary containing extraction results and file paths
        """
        symbol = symbol.upper().strip()
        output_dir = f"{symbol}_data"
        db_path = f"{output_dir}/{symbol}_stock_data.db"
        fii_dii_db_path = "fii_dii_trade.db"
        
        print(f"\n{'='*70}")
        print(f"STOCK DATA EXTRACTION FOR: {symbol}")
        print(f"{'='*70}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}/")
        
        # Initialize NSE India API with database
        nse = NseIndia(db_path=db_path)
        print(f"Stock Database: {db_path}")
        print(f"FII/DII Database: {fii_dii_db_path}")
        
        # Extract Layer 1 data (Symbol-specific)
        print("\n" + "-"*50)
        print("EXTRACTING LAYER 1: Symbol-Specific Data")
        print("-"*50)
        layer1_result = nse.extract_layer1_data(symbol)
        
        # Extract Layer 2 data (Sectorial)
        print("\n" + "-"*50)
        print("EXTRACTING LAYER 2: Sectorial Data")
        print("-"*50)
        layer2_result = nse.extract_layer2_data(symbol, layer1_result["json_file"])
        
        # Extract Layer 3 data (Index - NIFTY 50)
        print("\n" + "-"*50)
        print("EXTRACTING LAYER 3: Market Index Data")
        print("-"*50)
        layer3_result = nse.extract_layer3_data(symbol, fii_dii_db_path)
        
        # Generate Trading Signals from database
        print("\n" + "-"*50)
        print("GENERATING TRADING SIGNALS")
        print("-"*50)
        signal_file = nse.generate_trading_signals(symbol)
        
        # Close database connection
        if nse.db:
            nse.db.close()
        
        print(f"\n{'='*70}")
        print("DATA EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}/")
        print(f"Compiled JSON: {signal_file}")
        
        return {
            "symbol": symbol,
            "output_dir": output_dir,
            "compiled_file": signal_file,
            "layer1_result": layer1_result,
            "layer2_result": layer2_result,
            "layer3_result": layer3_result
        }
    
    def load_compiled_data(self, file_path: str) -> dict:
        """
        Load the compiled stock data from JSON file.
        
        Args:
            file_path: Path to the compiled JSON file
            
        Returns:
            Parsed dictionary of stock data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\nLoaded compiled data from: {file_path}")
            print(f"Data sections: {list(data.keys())}")
            return data
        except FileNotFoundError:
            print(f"Error: Compiled file not found at {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return None
    
    def prepare_llm_prompt(self, data: dict) -> str:
        """
        Prepare a comprehensive prompt for the LLM based on stock data.
        
        Args:
            data: Dictionary containing compiled stock signals
            
        Returns:
            Formatted prompt string for LLM analysis
        """
        symbol = data.get("introduction", {}).get("symbol", "UNKNOWN")
        company_name = data.get("introduction", {}).get("identity", {}).get("company_name", "Unknown Company")
        current_price = data.get("introduction", {}).get("current_price", {})
        
        # Extract key metrics
        trend = data.get("historical_signals", {}).get("trend_structure", {}).get("trend", "N/A")
        sma_200 = data.get("historical_signals", {}).get("trend_structure", {}).get("sma_200", 0)
        sma_50 = data.get("historical_signals", {}).get("trend_structure", {}).get("sma_50", 0)
        rsi = data.get("price_volume_signals", {}).get("technical_indicators", {}).get("rsi_14", 50)
        macd = data.get("price_volume_signals", {}).get("technical_indicators", {}).get("macd_histogram", 0)
        
        # Smart money signals
        smart_money = data.get("price_volume_signals", {}).get("smart_money_signals", {}).get("smart_money_signal", "N/A")
        delivery_trend = data.get("price_volume_signals", {}).get("smart_money_signals", {}).get("delivery_trend", "N/A")
        
        # Options data
        pcr = data.get("option_chain_signals", {}).get("market_expectation", {}).get("pcr_oi", 1.0)
        max_pain = data.get("option_chain_signals", {}).get("market_expectation", {}).get("max_pain_strike", 0)
        
        # Futures data
        futures_premium = data.get("futures_signals", {}).get("trend_direction", {}).get("futures_premium_pct", 0)
        oi_signal = data.get("futures_signals", {}).get("positioning_sentiment", {}).get("oi_signal", "N/A")
        
        # FII/DII flows
        fii_net = data.get("fii_dii_signals", {}).get("flow_bias", {}).get("fii_net_cr", 0)
        dii_net = data.get("fii_dii_signals", {}).get("flow_bias", {}).get("dii_net_cr", 0)
        
        # Sector performance
        sector_trend = data.get("sector_signals", {}).get("sector_trend_momentum", {}).get("trend", "N/A")
        stock_vs_sector = data.get("sector_signals", {}).get("relative_strength", {}).get("rs_signal", "N/A")
        
        # VIX for market sentiment
        vix = data.get("vix_history_signals", {}).get("volatility_regime", {}).get("current_vix", 20)
        vix_regime = data.get("vix_history_signals", {}).get("volatility_regime", {}).get("regime", "Normal")
        
        # Composite views from narrative
        intraday_view = data.get("narrative_summary", {}).get("intraday_view", {})
        swing_view = data.get("narrative_summary", {}).get("swing_view", {})
        positional_view = data.get("narrative_summary", {}).get("positional_view", {})
        
        intraday_bias = intraday_view.get("bias", "Neutral")
        swing_bias = swing_view.get("bias", "Neutral")
        positional_bias = positional_view.get("bias", "Neutral")
        
        overall_composite = data.get("narrative_summary", {}).get("overall_composite", {})
        overall_bias = overall_composite.get("overall_bias", "Neutral")
        
        # Fundamentals
        fundamentals = data.get("stock_details_signals", {})
        pe_ratio = fundamentals.get("valuation_metrics", {}).get("pe_ttm", "N/A")
        profit_growth = data.get("fundamental_signals", {}).get("earnings_profitability", {}).get("yoy_profit_growth_pct", 0)
        
        # Analyst recommendations
        analyst = fundamentals.get("analyst_recommendations", {})
        consensus = analyst.get("consensus", "N/A")
        buy_pct = analyst.get("buy_pct", 0)
        
        prompt = f"""You are an expert stock market analyst and investment advisor with 20+ years of experience in Indian stock markets (NSE/BSE). 

Please provide a comprehensive investment recommendation for {company_name} ({symbol}) based on the following multi-dimensional analysis:

## COMPANY OVERVIEW
- Company: {company_name}
- Symbol: {symbol}
- Current Trend: {trend}
- VIX Level: {vix} ({vix_regime} regime)

## TECHNICAL ANALYSIS
| Indicator | Value | Signal |
|-----------|-------|--------|
| Price vs SMA 200 | {'Above' if sma_200 > 0 else 'N/A'} | {'Bullish' if sma_200 > 0 else 'N/A'} |
| Price vs SMA 50 | {'Above' if sma_50 > 0 else 'N/A'} | {'Bullish' if sma_50 > 0 else 'N/A'} |
| RSI (14) | {rsi} | {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'} |
| MACD Histogram | {macd:.2f} | {'Bullish' if macd > 0 else 'Bearish'} |

## SMART MONEY INDICATORS
- Smart Money Signal: {smart_money}
- Delivery Trend: {delivery_trend}

## OPTIONS MARKET
- Put-Call Ratio (PCR): {pcr:.2f}
- Maximum Pain Strike: {max_pain}
- PCR Interpretation: {'Bullish' if pcr > 1.0 else 'Bearish' if pcr < 0.8 else 'Neutral'}

## FUTURES & OPTIONS POSITIONING
- Futures Premium: {futures_premium:.2f}%
- Open Interest Signal: {oi_signal}

## INSTITUTIONAL FLOW (FII/DII)
- FII Net Flow: ₹{fii_net:+.2f} Cr
- DII Net Flow: ₹{dii_net:+.2f} Cr
- Combined Institutional Sentiment: {'Bullish' if (fii_net + dii_net) > 0 else 'Bearish'}

## SECTOR ANALYSIS
- Sector Trend: {sector_trend}
- Stock vs Sector: {stock_vs_sector}

## FUNDAMENTALS
- P/E Ratio: {pe_ratio}
- YoY Profit Growth: {profit_growth:+.2f}%
- Analyst Consensus: {consensus}
- Analyst Buy %: {buy_pct}%

## MULTI-TIMEFRAME COMPOSITE ANALYSIS
| Timeframe | Bias |
|-----------|------|
| Intraday (0-1 Day) | {intraday_bias} |
| Swing (2-14 Days) | {swing_bias} |
| Positional (15+ Days) | {positional_bias} |
| OVERALL COMPOSITE | {overall_bias} |

## YOUR TASK

Based on ALL the above data, provide a clear investment recommendation in ONE of these four categories:

1. **BUY** - Strong buy recommendation with high confidence
2. **SELL** - Strong sell recommendation with high confidence  
3. **HOLD** - Hold existing positions, avoid new entries
4. **WAIT** - Stay on sidelines, await better entry point

Your analysis must include:
1. Your primary recommendation (BUY/SELL/HOLD/WAIT)
2. 3-5 key supporting reasons with specific data points
3. 2-3 risk factors or warning signs
4. Suggested stop-loss levels (if applicable)
5. Target price range with time horizon
6. Position sizing guidance (conservative/moderate/aggressive)

Please be decisive but balanced. Use proper formatting with headers and bullet points for readability. Focus on actionable insights rather than general market commentary.

Begin your analysis now:"""
        
        return prompt
    
    def get_llm_recommendation(self, prompt: str) -> str:
        """
        Send the prepared prompt to LLM and get streaming recommendation.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Streaming response from LLM
        """
        if not self.client:
            if not self.initialize_llm_client():
                return "Error: LLM client not available. Please check your API key."
        
        print(f"\n{'='*70}")
        print("SENDING DATA TO LLM FOR ANALYSIS")
        print(f"{'='*70}")
        print("\nAnalyzing stock data... (This may take a moment)\n")
        print("-"*70)
        print("LLM ANALYSIS AND RECOMMENDATION:")
        print("-"*70)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
                max_tokens=8192,
                stream=True
            )
            
            full_response = ""
            for chunk in completion:
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    print(reasoning, end="", flush=True)
                    full_response += reasoning
                
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n" + "-"*70)
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error getting LLM recommendation: {e}"
            print(error_msg)
            return error_msg
    
    def extract_recommendation(self, response: str) -> str:
        """
        Extract the clear recommendation from LLM response.
        
        Args:
            response: Full LLM response string
            
        Returns:
            Extracted recommendation category
        """
        response_upper = response.upper()
        
        if "**BUY**" in response or "\nBUY\n" in response or response_upper.count("BUY") > response_upper.count("SELL"):
            if response_upper.count("BUY") > response_upper.count("SELL") + 2:
                return "BUY"
        if "**SELL**" in response or "\nSELL\n" in response:
            return "SELL"
        if "**HOLD**" in response or "\nHOLD\n" in response:
            return "HOLD"
        if "**WAIT**" in response or "\nWAIT\n" in response:
            return "WAIT"
        
        # Fallback: look for recommendation in first 500 chars
        first_part = response[:500].upper()
        if "BUY" in first_part and "SELL" not in first_part:
            return "BUY"
        elif "SELL" in first_part:
            return "SELL"
        elif "HOLD" in first_part:
            return "HOLD"
        elif "WAIT" in first_part:
            return "WAIT"
        
        return "NEUTRAL"
    
    def run(self, symbol: str = None):
        """
        Main execution flow for stock recommendation.
        
        Args:
            symbol: Optional stock symbol. If None, will prompt user.
        """
        print("\n" + "="*70)
        print("     LLM-POWERED STOCK RECOMMENDATION SYSTEM")
        print("     Powered by NVIDIA API + Moonshot AI Kimi-k2-thinking")
        print("="*70)
        
        # Get stock symbol
        if not symbol:
            symbol = input("\nEnter stock ticker symbol (e.g., TCS, RELIANCE, INFY): ").strip()
        
        if not symbol:
            print("Error: No symbol provided!")
            return
        
        # Step 1: Get stock data
        extraction_result = self.get_stock_data(symbol)
        
        # Step 2: Load compiled data
        compiled_file = extraction_result.get("compiled_file")
        if not compiled_file or not os.path.exists(compiled_file):
            print(f"Error: Compiled file not found at {compiled_file}")
            return
        
        data = self.load_compiled_data(compiled_file)
        if not data:
            return
        
        # Step 3: Prepare LLM prompt
        print("\n" + "-"*50)
        print("PREPARING LLM PROMPT")
        print("-"*50)
        prompt = self.prepare_llm_prompt(data)
        print("Prompt prepared successfully!")
        
        # Step 4: Get LLM recommendation
        response = self.get_llm_recommendation(prompt)
        
        # Step 5: Extract and display recommendation
        recommendation = self.extract_recommendation(response)
        
        print("\n" + "="*70)
        print("FINAL RECOMMENDATION SUMMARY")
        print("="*70)
        print(f"\nStock: {symbol}")
        print(f"Recommendation: **{recommendation}**")
        print("\n" + "="*70)
        
        # Save recommendation to file
        recommendation_file = f"{extraction_result['output_dir']}/{symbol}_recommendation.txt"
        with open(recommendation_file, 'w', encoding='utf-8') as f:
            f.write(f"Stock Recommendation Report for {symbol}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            f.write(response)
            f.write(f"\n\nExtracted Recommendation: {recommendation}\n")
        
        print(f"\nRecommendation saved to: {recommendation_file}")
        
        return {
            "symbol": symbol,
            "recommendation": recommendation,
            "full_response": response,
            "output_dir": extraction_result["output_dir"],
            "recommendation_file": recommendation_file
        }


def main():
    """Main entry point for the stock recommendation system."""
    
    # Check for API key
    api_key = os.environ.get("NVIDIA_API_KEY")
    
    # Create recommender instance
    recommender = LLMStockRecommender(api_key=api_key)
    
    # Get symbol from command line arguments or user input
    symbol = None
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    
    # Run the recommendation process
    result = recommender.run(symbol=symbol)
    
    return result


if __name__ == "__main__":
    main()
