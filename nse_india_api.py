"""
NSE India API - Python implementation
Multi-layer data extraction for stock analysis
"""

import requests
import json
import time
import os
import csv
import sqlite3
import http.client
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import quote
from signal_generator import SignalGenerator


class StockDatabase:
    """SQLite database for storing and retrieving stock data"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._create_tables()
    
    def _create_tables(self):
        """Create all required tables"""
        cursor = self.conn.cursor()
        
        # Stock info table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol TEXT PRIMARY KEY,
                company_name TEXT,
                industry TEXT,
                isin TEXT,
                listing_date TEXT,
                face_value REAL,
                pd_sector_ind TEXT,
                pd_sector_ind_all TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Layer 1 JSON data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS layer1_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                data_type TEXT,
                json_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, data_type)
            )
        ''')
        
        # Layer 2 sectorial data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS layer2_sectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                sector_name TEXT,
                yahoo_symbol TEXT,
                json_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, sector_name)
            )
        ''')
        
        # Price/Volume data (from CSV)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_volume_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                trade_date TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                delivery_qty REAL,
                delivery_pct REAL,
                data_source TEXT,
                UNIQUE(symbol, trade_date, data_source)
            )
        ''')
        
        # Futures data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS futures_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                trade_date TEXT,
                expiry_date TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                open_interest REAL,
                change_in_oi REAL,
                UNIQUE(symbol, trade_date, expiry_date)
            )
        ''')
        
        # Option chain data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_chain_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                expiry_date TEXT,
                strike_price REAL,
                option_type TEXT,
                open_interest REAL,
                change_in_oi REAL,
                volume REAL,
                iv REAL,
                ltp REAL,
                bid_price REAL,
                ask_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, expiry_date, strike_price, option_type)
            )
        ''')
        
        # Sector index history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sector_index_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_name TEXT,
                trade_date TEXT,
                open_value REAL,
                high_value REAL,
                low_value REAL,
                close_value REAL,
                volume REAL,
                turnover REAL,
                UNIQUE(index_name, trade_date)
            )
        ''')
        
        # Layer 3: Index history data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_name TEXT,
                trade_date TEXT,
                open_value REAL,
                high_value REAL,
                low_value REAL,
                close_value REAL,
                volume REAL,
                turnover REAL,
                UNIQUE(index_name, trade_date)
            )
        ''')
        
        # Layer 3: VIX history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vix_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT,
                open_value REAL,
                high_value REAL,
                low_value REAL,
                close_value REAL,
                prev_close REAL,
                change_value REAL,
                change_pct REAL,
                UNIQUE(trade_date)
            )
        ''')
        
        # Layer 3: Bulk deals (filtered by symbol)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bulk_deals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                deal_date TEXT,
                client_name TEXT,
                deal_type TEXT,
                quantity REAL,
                price REAL,
                UNIQUE(symbol, deal_date, client_name, deal_type)
            )
        ''')
        
        # Layer 3: Block deals (filtered by symbol)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS block_deals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                deal_date TEXT,
                client_name TEXT,
                deal_type TEXT,
                quantity REAL,
                price REAL,
                UNIQUE(symbol, deal_date, client_name, deal_type)
            )
        ''')
        
        # Layer 3: Index futures data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_futures_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_name TEXT,
                trade_date TEXT,
                expiry_date TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                open_interest REAL,
                change_in_oi REAL,
                UNIQUE(index_name, trade_date, expiry_date)
            )
        ''')
        
        # Layer 3: JSON data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS layer3_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                data_type TEXT,
                json_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, data_type)
            )
        ''')
        
        # Layer 3: Index option chain
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_option_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_name TEXT,
                expiry_date TEXT,
                strike_price REAL,
                option_type TEXT,
                open_interest REAL,
                change_in_oi REAL,
                volume REAL,
                iv REAL,
                ltp REAL,
                bid_price REAL,
                ask_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(index_name, expiry_date, strike_price, option_type)
            )
        ''')
        
        self.conn.commit()
    
    def save_stock_info(self, symbol, data):
        """Save basic stock info"""
        cursor = self.conn.cursor()
        
        info = data.get("info", {})
        metadata = data.get("metadata", {})
        security_info = data.get("securityInfo", {})
        
        pd_sector_ind_all = metadata.get("pdSectorIndAll", [])
        if isinstance(pd_sector_ind_all, list):
            pd_sector_ind_all = json.dumps(pd_sector_ind_all)
        
        cursor.execute('''
            INSERT OR REPLACE INTO stock_info 
            (symbol, company_name, industry, isin, listing_date, face_value, 
             pd_sector_ind, pd_sector_ind_all, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            symbol,
            info.get("companyName", ""),
            info.get("industry", ""),
            info.get("isin", ""),
            info.get("listingDate", ""),
            security_info.get("faceValue", 0),
            metadata.get("pdSectorInd", ""),
            pd_sector_ind_all
        ))
        self.conn.commit()
    
    def save_layer1_json(self, symbol, data_type, json_data):
        """Save Layer 1 JSON data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO layer1_data (symbol, data_type, json_data)
            VALUES (?, ?, ?)
        ''', (symbol, data_type, json.dumps(json_data) if not isinstance(json_data, str) else json_data))
        self.conn.commit()
    
    def save_layer2_sector(self, symbol, sector_name, yahoo_symbol, json_data):
        """Save Layer 2 sector data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO layer2_sectors (symbol, sector_name, yahoo_symbol, json_data)
            VALUES (?, ?, ?, ?)
        ''', (symbol, sector_name, yahoo_symbol, json.dumps(json_data) if not isinstance(json_data, str) else json_data))
        self.conn.commit()
    
    def save_layer3_json(self, symbol, data_type, json_data):
        """Save Layer 3 JSON data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO layer3_data (symbol, data_type, json_data)
            VALUES (?, ?, ?)
        ''', (symbol, data_type, json.dumps(json_data) if not isinstance(json_data, str) else json_data))
        self.conn.commit()
    
    def get_stock_info(self, symbol):
        """Get stock info"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM stock_info WHERE symbol = ?', (symbol,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_pd_sector_ind_all(self, symbol):
        """Get pdSectorIndAll list for a symbol"""
        info = self.get_stock_info(symbol)
        if info and info.get("pd_sector_ind_all"):
            try:
                return json.loads(info["pd_sector_ind_all"])
            except:
                return []
        return []
    
    def get_layer1_data(self, symbol, data_type=None):
        """Get Layer 1 data"""
        cursor = self.conn.cursor()
        if data_type:
            cursor.execute('SELECT * FROM layer1_data WHERE symbol = ? AND data_type = ?', (symbol, data_type))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                try:
                    result['json_data'] = json.loads(result['json_data'])
                except:
                    pass
                return result
        else:
            cursor.execute('SELECT * FROM layer1_data WHERE symbol = ?', (symbol,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        return None
    
    def _parse_date_for_sort(self, date_str):
        """Parse date string for sorting"""
        if not date_str:
            return (0, 0, 0)
        date_str = str(date_str).strip()
        formats = ["%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%B-%Y"]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return (dt.year, dt.month, dt.day)
            except:
                continue
        return (0, 0, 0)
    
    def save_sector_index_history(self, index_name, records):
        """Save sector index history records to database - sorted by date (oldest first)"""
        # Sort records by date ascending
        sorted_records = sorted(records, key=lambda x: self._parse_date_for_sort(
            x.get("EOD_TIMESTAMP", x.get("Date", x.get("date", "")))
        ))
        
        cursor = self.conn.cursor()
        for rec in sorted_records:
            cursor.execute('''
                INSERT OR REPLACE INTO sector_index_history 
                (index_name, trade_date, open_value, high_value, low_value, close_value, volume, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                index_name,
                rec.get("EOD_TIMESTAMP", rec.get("Date", rec.get("date", ""))),
                rec.get("EOD_OPEN_INDEX_VAL", rec.get("Open", rec.get("open", 0))),
                rec.get("EOD_HIGH_INDEX_VAL", rec.get("High", rec.get("high", 0))),
                rec.get("EOD_LOW_INDEX_VAL", rec.get("Low", rec.get("low", 0))),
                rec.get("EOD_CLOSE_INDEX_VAL", rec.get("Close", rec.get("close", 0))),
                rec.get("HIT_TRADED_QTY", rec.get("EOD_INDEX_TRADING_VOL", rec.get("Volume", 0))),
                rec.get("HIT_TURN_OVER", rec.get("EOD_INDEX_TRADING_TURNOVER", rec.get("Turnover", 0)))
            ))
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class FiiDiiDatabase:
    """Separate database for FII/DII trade data - updated daily at 3:30 PM IST"""
    
    def __init__(self, db_path="fii_dii_trade.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create FII/DII tables"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fii_dii_trade (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT,
                category TEXT,
                buy_value REAL,
                sell_value REAL,
                net_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(trade_date, category)
            )
        ''')
        
        self.conn.commit()
    
    def save_fii_dii_data(self, data):
        """Save FII/DII trade data - API returns list directly"""
        cursor = self.conn.cursor()
        count = 0
        
        # API returns list: [{"category":"DII","date":"02-Jan-2026",...}, {...}]
        records = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        
        for category_data in records:
            trade_date = category_data.get("date", datetime.now().strftime("%d-%b-%Y"))
            category = category_data.get("category", "")
            
            if not category:
                continue
                
            cursor.execute('''
                INSERT OR REPLACE INTO fii_dii_trade 
                (trade_date, category, buy_value, sell_value, net_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                trade_date,
                category,
                float(str(category_data.get("buyValue", 0)).replace(",", "") or 0),
                float(str(category_data.get("sellValue", 0)).replace(",", "") or 0),
                float(str(category_data.get("netValue", 0)).replace(",", "") or 0)
            ))
            count += 1
        self.conn.commit()
        return count
    
    def get_all_data(self):
        """Get all FII/DII data"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM fii_dii_trade ORDER BY trade_date DESC')
        return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_data(self):
        """Get latest FII/DII data"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM fii_dii_trade ORDER BY trade_date DESC LIMIT 10')
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        self.conn.close()


class NseIndia:
    def __init__(self, db_path=None):
        self.base_url = "https://www.nseindia.com"
        self.session = requests.Session()
        self.cookies_set = False
        self.indian_api_key = "sk-live-c1IGHHZptRhUUhpvrGucCUrqXXiEZEDRPE344kYi"
        self.db = None
        
        if db_path:
            self.db = StockDatabase(db_path)
        
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://www.nseindia.com/",
        })

    def _init_session(self):
        if self.cookies_set:
            return
        
        print("Initializing NSE session...")
        self.session.get(self.base_url, timeout=30)
        time.sleep(1)
        self.session.get(f"{self.base_url}/get-quotes/equity?symbol=TCS", timeout=30)
        time.sleep(1)
        self.cookies_set = True
        print("Session ready")

    def get_data(self, url):
        """Get raw text data from URL"""
        self._init_session()
        time.sleep(0.5)
        response = self.session.get(url, timeout=30)
        return response.text

    def get_json(self, url):
        """Get data and parse as JSON"""
        text = self.get_data(url)
        try:
            return json.loads(text)
        except:
            return None

    # ==================== LAYER 1: Symbol-Specific APIs ====================
    
    def get_equity_details(self, symbol):
        """Get equity quote details - returns JSON text"""
        url = f"{self.base_url}/api/quote-equity?symbol={quote(symbol.upper())}"
        return self.get_data(url)

    def get_equity_trade_info(self, symbol):
        """Get equity trade information - returns JSON text"""
        url = f"{self.base_url}/api/quote-equity?symbol={quote(symbol.upper())}&section=trade_info"
        return self.get_data(url)

    def get_equity_corporate_info(self, symbol):
        """Get corporate information - returns JSON text"""
        url = f"{self.base_url}/api/top-corp-info?symbol={quote(symbol.upper())}&market=equities"
        return self.get_data(url)

    def get_equity_intraday_data(self, symbol):
        """Get intraday chart data - returns JSON text"""
        details_url = f"{self.base_url}/api/quote-equity?symbol={quote(symbol.upper())}"
        details = self.get_json(details_url)
        
        if details and "info" in details:
            identifier = details["info"]["identifier"]
            url = (f"{self.base_url}/api/NextApi/apiClient/GetQuoteApi?"
                   f"functionName=getSymbolChartData&symbol={quote(identifier)}&days=1D")
            return self.get_data(url)
        return '{"error": "Could not get identifier"}'

    def get_equity_historical_data(self, symbol, from_date=None, to_date=None):
        """Get historical trade data - returns JSON text"""
        symbol = symbol.upper()
        details_url = f"{self.base_url}/api/quote-equity?symbol={quote(symbol)}"
        details = self.get_json(details_url)
        
        series = "EQ"
        if details and "info" in details:
            active_series = details["info"].get("activeSeries", [])
            if active_series:
                series = active_series[0]
        
        if not to_date:
            to_date = datetime.now().strftime("%d-%m-%Y")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        
        url = (f"{self.base_url}/api/NextApi/apiClient/GetQuoteApi?"
               f"functionName=getHistoricalTradeData"
               f"&symbol={quote(symbol)}"
               f"&series={quote(series)}"
               f"&fromDate={from_date}&toDate={to_date}")
        return self.get_data(url)

    def get_results_comparison(self, symbol):
        """Get results comparison - returns JSON text"""
        url = f"{self.base_url}/api/results-comparision?symbol={quote(symbol.upper())}"
        return self.get_data(url)

    # ==================== NEW LAYER 1 APIs ====================

    def get_price_volume_deliverable_data(self, symbol, output_dir):
        """
        Get price volume and deliverable position data for last 5 years (1 year at a time)
        Returns CSV data - saves to files
        Useful for: Total traded quantity, Delivery quantity, Delivery %, Volume trend
        """
        symbol = symbol.upper()
        all_csv_files = []
        
        today = datetime.now()
        
        for year_offset in range(5):
            to_date = today - timedelta(days=365 * year_offset)
            from_date = to_date - timedelta(days=365)
            
            from_str = from_date.strftime("%d-%m-%Y")
            to_str = to_date.strftime("%d-%m-%Y")
            
            print(f"  Fetching price/volume/deliverable data: {from_str} to {to_str}...")
            
            url = (f"{self.base_url}/api/historicalOR/generateSecurityWiseHistoricalData?"
                   f"from={from_str}&to={to_str}&symbol={quote(symbol)}"
                   f"&type=priceVolumeDeliverable&series=ALL&csv=true")
            
            try:
                csv_data = self.get_data(url)
                
                filename = f"{output_dir}/{symbol}_price_volume_deliverable_year{year_offset + 1}.csv"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(csv_data)
                all_csv_files.append(filename)
                print(f"    Saved: {filename}")
            except Exception as e:
                print(f"    Error for year {year_offset + 1}: {e}")
            
            time.sleep(1)
        
        return all_csv_files

    def get_future_price_volume_data(self, symbol, output_dir):
        """
        Get futures price volume data for last 1 year
        Converts tabular JSON data to CSV
        Useful for: Futures price (all expiries), Open Interest, Change in OI, Volume
        """
        symbol = symbol.upper()
        
        today = datetime.now()
        from_date = today - timedelta(days=365)
        
        from_str = from_date.strftime("%d-%m-%Y")
        to_str = today.strftime("%d-%m-%Y")
        
        print(f"  Fetching futures data: {from_str} to {to_str}...")
        
        # Fetch without csv=true to get JSON tabular data
        url = (f"{self.base_url}/api/historicalOR/foCPV?"
               f"from={from_str}&to={to_str}"
               f"&instrumentType=FUTSTK&symbol={quote(symbol)}")
        
        try:
            raw_data = self.get_data(url)
            filename = f"{output_dir}/{symbol}_futures_price_volume.csv"
            
            # Try to parse as JSON and convert to CSV
            try:
                json_data = json.loads(raw_data)
                
                # Handle different JSON structures
                if isinstance(json_data, list) and len(json_data) > 0:
                    # List of records
                    records = json_data
                elif isinstance(json_data, dict):
                    # Check for common data keys
                    if "data" in json_data:
                        records = json_data["data"]
                    elif "records" in json_data:
                        records = json_data["records"]
                    else:
                        # Flatten single dict to list
                        records = [json_data]
                else:
                    records = []
                
                if records and isinstance(records, list) and len(records) > 0:
                    # Write as CSV
                    fieldnames = records[0].keys() if isinstance(records[0], dict) else []
                    with open(filename, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for record in records:
                            if isinstance(record, dict):
                                writer.writerow(record)
                    print(f"    Saved (JSON→CSV): {filename}")
                    return filename
                else:
                    # No records, save raw
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(raw_data)
                    print(f"    Saved (raw): {filename}")
                    return filename
                    
            except json.JSONDecodeError:
                # Not JSON, might already be CSV format
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(raw_data)
                print(f"    Saved (raw CSV): {filename}")
                return filename
                
        except Exception as e:
            print(f"    Error: {e}")
            return None

    def get_option_chain_all_expiries(self, symbol, output_dir):
        """
        Get option chain for all expiry dates
        First fetches expiry dates, then gets option chain for each expiry
        Converts tabular data to CSV
        Useful for: Strike-wise OI, OI change, PCR, Option volume
        """
        symbol = symbol.upper()
        
        # Step 1: Get expiry dates
        print(f"  Fetching expiry dates for {symbol}...")
        contract_info_url = f"{self.base_url}/api/option-chain-contract-info?symbol={quote(symbol)}"
        contract_info = self.get_json(contract_info_url)
        
        if not contract_info or "expiryDates" not in contract_info:
            print("    Could not fetch expiry dates")
            return []
        
        expiry_dates = contract_info.get("expiryDates", [])
        print(f"    Found {len(expiry_dates)} expiry dates")
        
        all_csv_files = []
        
        # Step 2: Get option chain for each expiry
        for expiry in expiry_dates:
            print(f"    Fetching option chain for expiry: {expiry}...")
            
            url = (f"{self.base_url}/api/option-chain-v3?type=Equity"
                   f"&symbol={quote(symbol)}&expiry={quote(expiry)}")
            
            try:
                option_data = self.get_json(url)
                
                if option_data and "records" in option_data and "data" in option_data["records"]:
                    records = option_data["records"]["data"]
                    
                    # Convert to CSV
                    expiry_clean = expiry.replace("-", "_")
                    filename = f"{output_dir}/{symbol}_option_chain_{expiry_clean}.csv"
                    
                    if records:
                        # Flatten the nested structure
                        csv_rows = []
                        for record in records:
                            row = {
                                "strikePrice": record.get("strikePrice", ""),
                                "expiryDate": record.get("expiryDate", ""),
                            }
                            # CE data
                            ce = record.get("CE", {})
                            for key, val in ce.items():
                                row[f"CE_{key}"] = val
                            # PE data
                            pe = record.get("PE", {})
                            for key, val in pe.items():
                                row[f"PE_{key}"] = val
                            csv_rows.append(row)
                        
                        if csv_rows:
                            fieldnames = csv_rows[0].keys()
                            with open(filename, "w", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(csv_rows)
                            all_csv_files.append(filename)
                            print(f"      Saved: {filename}")
                else:
                    print(f"      No data for expiry {expiry}")
                    
            except Exception as e:
                print(f"      Error for expiry {expiry}: {e}")
            
            time.sleep(0.5)
        
        return all_csv_files

    def get_stock_details_indianapi(self, symbol):
        """
        Get complete stock details from indianapi.in
        Returns JSON data
        """
        print(f"  Fetching stock details from IndianAPI...")
        
        try:
            conn = http.client.HTTPSConnection("stock.indianapi.in")
            headers = {"X-Api-Key": self.indian_api_key}
            conn.request("GET", f"/stock?name={quote(symbol.upper())}", headers=headers)
            
            res = conn.getresponse()
            data = res.read()
            conn.close()
            
            return data.decode("utf-8")
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_historical_stats_indianapi(self, symbol):
        """
        Get complete historical state of stock from indianapi.in
        Returns JSON data
        """
        print(f"  Fetching historical stats from IndianAPI...")
        
        try:
            conn = http.client.HTTPSConnection("stock.indianapi.in")
            headers = {"X-Api-Key": self.indian_api_key}
            conn.request("GET", f"/historical_stats?stock_name={quote(symbol.upper())}&stats=quarter_results", headers=headers)
            
            res = conn.getresponse()
            data = res.read()
            conn.close()
            
            return data.decode("utf-8")
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ==================== Market-wide APIs ====================
    
    def get_market_status(self):
        """Get market status"""
        return self.get_data(f"{self.base_url}/api/marketStatus")

    def get_fii_dii_trade_react(self):
        """Get FII/DII trading activity"""
        return self.get_data(f"{self.base_url}/api/fiidiiTradeReact")

    def get_index_intraday_data(self, index):
        """Get index intraday chart data"""
        url = (f"{self.base_url}/api/NextApi/apiClient?functionName=getGraphChart"
               f"&type={quote(index.upper())}&flag=1D")
        return self.get_data(url)

    def get_index_option_chain_contract_info(self, index_symbol):
        """Get option chain contract info for index"""
        url = f"{self.base_url}/api/option-chain-contract-info?symbol={quote(index_symbol.upper())}"
        return self.get_data(url)

    # ==================== LAYER 3: Index Data APIs ====================
    
    def get_index_history_data(self, index_name="NIFTY 50", from_date=None, to_date=None):
        """
        Get index historical data for 1 year
        Useful for: Total traded quantity, Volume trend, turnover, index values, Trend & momentum
        """
        if not to_date:
            to_date = datetime.now().strftime("%d-%m-%Y")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        
        url = (f"{self.base_url}/api/historicalOR/indicesHistory?"
               f"indexType={quote(index_name)}&from={from_date}&to={to_date}")
        return self.get_data(url)
    
    def get_nse_index_option_chain(self, index_symbol="NIFTY", output_dir=None):
        """
        Get option chain for all expiry dates for an index
        Useful for: Strike-wise OI, OI change, PCR, Option volume, Smart money positioning
        """
        index_symbol = index_symbol.upper()
        
        # Step 1: Get expiry dates
        print(f"  Fetching expiry dates for {index_symbol}...")
        contract_info = self.get_json(f"{self.base_url}/api/option-chain-contract-info?symbol={quote(index_symbol)}")
        
        if not contract_info or "expiryDates" not in contract_info:
            print("    Could not fetch expiry dates")
            return []
        
        expiry_dates = contract_info.get("expiryDates", [])
        print(f"    Found {len(expiry_dates)} expiry dates")
        
        all_csv_files = []
        all_records = []
        
        # Step 2: Get option chain for each expiry
        for expiry in expiry_dates:
            print(f"    Fetching option chain for expiry: {expiry}...")
            
            url = f"{self.base_url}/api/option-chain-v3?type=Indices&symbol={quote(index_symbol)}&expiry={quote(expiry)}"
            
            try:
                option_data = self.get_json(url)
                
                if option_data and "records" in option_data and "data" in option_data["records"]:
                    records = option_data["records"]["data"]
                    
                    if records and output_dir:
                        # Flatten and save to CSV
                        csv_rows = []
                        for record in records:
                            row = {
                                "strikePrice": record.get("strikePrice", ""),
                                "expiryDate": record.get("expiryDate", expiry),
                            }
                            ce = record.get("CE", {})
                            for key, val in ce.items():
                                row[f"CE_{key}"] = val
                            pe = record.get("PE", {})
                            for key, val in pe.items():
                                row[f"PE_{key}"] = val
                            csv_rows.append(row)
                            all_records.append(row)
                        
                        if csv_rows:
                            expiry_clean = expiry.replace("-", "_")
                            filename = f"{output_dir}/{index_symbol}_index_option_chain_{expiry_clean}.csv"
                            fieldnames = csv_rows[0].keys()
                            with open(filename, "w", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(csv_rows)
                            all_csv_files.append(filename)
                            print(f"      Saved: {filename}")
                else:
                    print(f"      No data for expiry {expiry}")
                    
            except Exception as e:
                print(f"      Error for expiry {expiry}: {e}")
            
            time.sleep(0.5)
        
        return all_csv_files, all_records
    
    def get_india_vix_data(self, from_date=None, to_date=None):
        """
        Get India VIX historical data for 1 year
        Useful for: Market volatility check
        """
        if not to_date:
            to_date = datetime.now().strftime("%d-%m-%Y")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        
        url = f"{self.base_url}/api/historicalOR/vixhistory?from={from_date}&to={to_date}&csv=true"
        return self.get_data(url)
    
    def get_bulk_deals_data(self, from_date=None, to_date=None):
        """
        Get bulk deals data for 1 year
        """
        if not to_date:
            to_date = datetime.now().strftime("%d-%m-%Y")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        
        url = f"{self.base_url}/api/historicalOR/bulk-block-short-deals?optionType=bulk_deals&from={from_date}&to={to_date}&csv=true"
        return self.get_data(url)
    
    def get_block_deals_data(self, from_date=None, to_date=None):
        """
        Get block deals data for 1 year
        """
        if not to_date:
            to_date = datetime.now().strftime("%d-%m-%Y")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        
        url = f"{self.base_url}/api/historicalOR/bulk-block-short-deals?optionType=block_deals&from={from_date}&to={to_date}&csv=true"
        return self.get_data(url)
    
    def get_index_future_price_volume_data(self, index_symbol="NIFTY", from_date=None, to_date=None):
        """
        Get index futures price volume data for 3 months
        Useful for: Futures price, Open Interest, Change in OI, Volume, Long/short buildup
        """
        if not to_date:
            to_date = datetime.now().strftime("%d-%m-%Y")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=90)).strftime("%d-%m-%Y")
        
        url = (f"{self.base_url}/api/historicalOR/foCPV?"
               f"from={from_date}&to={to_date}&instrumentType=FUTIDX&symbol={quote(index_symbol.upper())}&csv=true")
        return self.get_data(url)

    # ==================== LAYER 2: Sectorial Data APIs ====================
    
    # Mapping: NSE Index Name -> Yahoo Finance Symbol
    SECTOR_INDEX_MAPPING = {
        "NIFTY AUTO": "^CNXAUTO",
        "NIFTY CHEMICALS": "NIFTY_CHEMICALS.NS",
        "NIFTY CONSUMER DURABLES": "NIFTY_CONSR_DURBL.NS",
        "NIFTY FINANCIAL SERVICES EX-BANK": "NIFTY_FIN_SERVICE.NS",
        "NIFTY FINANCIAL SERVICES 25/50": "^CNXFIN",
        "NIFTY FMCG": "^CNXFMCG",
        "NIFTY HEALTHCARE INDEX": "NIFTY_HEALTHCARE.NS",
        "NIFTY IT": "^CNXIT",
        "NIFTY MEDIA": "^CNXMEDIA",
        "NIFTY METAL": "^CNXMETAL",
        "NIFTY OIL & GAS": "NIFTY_OIL_AND_GAS.NS",
        "NIFTY PHARMA": "^CNXPHARMA",
        "NIFTY PSU BANK": "^CNXPSUBANK",
        "NIFTY REALTY": "^CNXREALTY",
    }

    def get_all_indices(self):
        """Get all indices data"""
        return self.get_data(f"{self.base_url}/api/allIndices")

    def get_sector_index_history(self, index_name, from_date=None, to_date=None):
        """
        Get historical data for a sector index (1 year)
        Useful for: Volume trend, turnover, index values, Trend & momentum
        """
        if not to_date:
            to_date = datetime.now().strftime("%d-%m-%Y")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        
        url = (f"{self.base_url}/api/historicalOR/indicesHistory?"
               f"indexType={quote(index_name)}&from={from_date}&to={to_date}")
        return self.get_data(url)

    def get_sector_index_intraday(self, index_name):
        """
        Get intraday data for a sector index
        Useful for: Intraday momentum
        Note: API may require specific case - trying uppercase first
        """
        # API expects "AND" instead of "&" in index names
        api_index_name = index_name.replace("&", "AND").upper()
        
        url = (f"{self.base_url}/api/NextApi/apiClient?functionName=getGraphChart"
               f"&type={quote(api_index_name)}&flag=1D")
        return self.get_data(url)

    def extract_matching_sectors(self, pd_sector_ind_all):
        """
        Match pdSectorIndAll list with known sector indices
        Returns list of matched indices with their Yahoo Finance symbols
        """
        matched = []
        for sector in pd_sector_ind_all:
            sector_upper = sector.upper().strip()
            for nse_name, yahoo_symbol in self.SECTOR_INDEX_MAPPING.items():
                if nse_name in sector_upper or sector_upper in nse_name:
                    matched.append({
                        "nse_name": nse_name,
                        "yahoo_symbol": yahoo_symbol,
                        "original": sector
                    })
                    break
        return matched

    def extract_layer2_data(self, symbol, layer1_json_file):
        """
        Extract Layer 2 (sectorial) data based on pdSectorIndAll from Layer 1
        """
        symbol = symbol.upper()
        output_dir = f"{symbol}_data"
        
        print(f"\n{'='*60}")
        print(f"LAYER 2: Extracting Sectorial Data for {symbol}")
        print(f"{'='*60}\n")
        
        # Try to get pdSectorIndAll from database first
        pd_sector_ind_all = []
        if self.db:
            print("Loading from database...")
            pd_sector_ind_all = self.db.get_pd_sector_ind_all(symbol)
        
        # Fallback to JSON file if database doesn't have data
        if not pd_sector_ind_all:
            print("Loading Layer 1 data from JSON file...")
            try:
                with open(layer1_json_file, "r", encoding="utf-8") as f:
                    layer1_data = json.load(f)
            except Exception as e:
                print(f"Error loading Layer 1 JSON: {e}")
                return None
            
            # Extract pdSectorIndAll from equity_details -> metadata
            equity_details = layer1_data.get("equity_details", {})
            if isinstance(equity_details, dict):
                # Primary path: metadata -> pdSectorIndAll
                pd_sector_ind_all = equity_details.get("metadata", {}).get("pdSectorIndAll", [])
                
                # Fallback paths
                if not pd_sector_ind_all:
                    pd_sector_ind_all = equity_details.get("info", {}).get("pdSectorIndAll", [])
                if not pd_sector_ind_all:
                    pd_sector_ind_all = equity_details.get("metadata", {}).get("pdSectorInd", [])
                    if pd_sector_ind_all and isinstance(pd_sector_ind_all, str):
                        pd_sector_ind_all = [pd_sector_ind_all]
        
        if not pd_sector_ind_all:
            print("No pdSectorIndAll found")
        
        print(f"Found pdSectorIndAll: {pd_sector_ind_all}")
        
        # Match with known sector indices
        matched_sectors = self.extract_matching_sectors(pd_sector_ind_all)
        print(f"Matched {len(matched_sectors)} sector indices:")
        for m in matched_sectors:
            print(f"  - {m['nse_name']} ({m['yahoo_symbol']})")
        
        layer2_data = {
            "symbol": symbol,
            "pd_sector_ind_all": pd_sector_ind_all,
            "matched_sectors": matched_sectors,
            "sector_data": {}
        }
        
        # 1. Get All Indices Data
        print("\n1. Fetching All Indices Data...")
        try:
            all_indices_raw = self.get_all_indices()
            all_indices = json.loads(all_indices_raw)
            
            # Filter for matched sectors
            if "data" in all_indices:
                filtered_indices = []
                for idx_data in all_indices["data"]:
                    idx_name = idx_data.get("index", "").upper()
                    for m in matched_sectors:
                        if m["nse_name"].upper() in idx_name or idx_name in m["nse_name"].upper():
                            filtered_indices.append(idx_data)
                            break
                layer2_data["all_indices_filtered"] = filtered_indices
                print(f"  Filtered {len(filtered_indices)} matching indices from all indices")
            else:
                layer2_data["all_indices_raw"] = all_indices
        except Exception as e:
            layer2_data["all_indices_error"] = str(e)
            print(f"  Error: {e}")
        
        # 2. Get Historical Data for each matched sector (CSV)
        print("\n2. Fetching Sector Index Historical Data (1 year)...")
        sector_history_files = []
        for m in matched_sectors:
            index_name = m["nse_name"]
            print(f"  Fetching history for {index_name}...")
            try:
                history_raw = self.get_sector_index_history(index_name)
                
                # Try to parse and convert to CSV
                try:
                    history_json = json.loads(history_raw)
                    records = []
                    if isinstance(history_json, dict) and "data" in history_json:
                        records = history_json["data"]
                    elif isinstance(history_json, list):
                        records = history_json
                    
                    if records:
                        clean_name = index_name.replace(" ", "_").replace("&", "AND")
                        filename = f"{output_dir}/{symbol}_sector_{clean_name}_history.csv"
                        
                        if isinstance(records[0], dict):
                            fieldnames = records[0].keys()
                            with open(filename, "w", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(records)
                            sector_history_files.append(filename)
                            print(f"    Saved: {filename}")
                            
                            # Save to database
                            if self.db:
                                self.db.save_sector_index_history(index_name, records)
                                print(f"    Saved to database: {len(records)} records")
                        else:
                            # Save raw if not dict records
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(history_raw)
                            sector_history_files.append(filename)
                    else:
                        layer2_data["sector_data"][f"{index_name}_history"] = {"raw": history_raw}
                except json.JSONDecodeError:
                    layer2_data["sector_data"][f"{index_name}_history"] = {"raw": history_raw}
                    
            except Exception as e:
                print(f"    Error: {e}")
                layer2_data["sector_data"][f"{index_name}_history_error"] = str(e)
            
            time.sleep(0.5)
        
        # 3. Get Intraday Data for each matched sector
        print("\n3. Fetching Sector Index Intraday Data...")
        for m in matched_sectors:
            index_name = m["nse_name"]
            print(f"  Fetching intraday for {index_name}...")
            try:
                intraday_raw = self.get_sector_index_intraday(index_name)
                try:
                    intraday_json = json.loads(intraday_raw)
                    layer2_data["sector_data"][f"{index_name}_intraday"] = intraday_json
                except:
                    layer2_data["sector_data"][f"{index_name}_intraday"] = {"raw": intraday_raw}
            except Exception as e:
                layer2_data["sector_data"][f"{index_name}_intraday_error"] = str(e)
                print(f"    Error: {e}")
            
            time.sleep(0.5)
        
        # Save Layer 2 JSON
        json_file = f"{output_dir}/{symbol}_layer2_data.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(layer2_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON data saved to: {json_file}")
        
        # Save to database if available
        if self.db:
            print("Saving sector data to database...")
            for m in matched_sectors:
                sector_data = layer2_data["sector_data"].get(f"{m['nse_name']}_intraday", {})
                self.db.save_layer2_sector(symbol, m["nse_name"], m["yahoo_symbol"], sector_data)
            print("  Sector data saved to SQLite database")
        
        # Summary
        print(f"\n{'='*60}")
        print("LAYER 2 EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Matched sectors: {len(matched_sectors)}")
        print(f"JSON file: {json_file}")
        print(f"History CSV files: {len(sector_history_files)}")
        
        return {
            "output_dir": output_dir,
            "json_file": json_file,
            "history_csv_files": sector_history_files,
            "matched_sectors": matched_sectors
        }

    # ==================== CSV TO DATABASE HELPERS ====================
    
    def _normalize_row(self, row):
        """Normalize CSV row - strip whitespace and BOM from keys"""
        normalized = {}
        for key, value in row.items():
            if key:
                # Remove BOM (various encodings), quotes, and strip whitespace
                clean_key = key.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
                clean_key = clean_key.replace('Ã¯Â»Â¿', '').replace('"', '').strip()
                normalized[clean_key] = str(value).strip() if value else ''
        return normalized
    
    def _get_value(self, row, *keys):
        """Get value from row trying multiple key names"""
        for key in keys:
            if key in row and row[key]:
                return row[key]
        return ''
    
    def _parse_float(self, value):
        """Parse float from string, handling commas and special chars"""
        if not value:
            return 0.0
        try:
            return float(str(value).replace(',', '').replace('%', '').replace('#', '').strip() or 0)
        except:
            return 0.0
    
    def _parse_date_for_sort(self, date_str):
        """Parse date string to sortable format. Returns tuple (year, month, day) for sorting."""
        if not date_str:
            return (0, 0, 0)
        date_str = str(date_str).strip()
        
        # Try different date formats (including 2-digit years)
        formats = [
            "%d-%b-%Y",    # 01-Jan-2025
            "%d-%b-%y",    # 01-Jan-25
            "%d-%m-%Y",    # 01-01-2025
            "%d-%m-%y",    # 01-01-25
            "%Y-%m-%d",    # 2025-01-01
            "%d/%m/%Y",    # 01/01/2025
            "%d/%m/%y",    # 01/01/25
            "%d-%B-%Y",    # 01-January-2025
            "%d-%B-%y",    # 01-January-25
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return (dt.year, dt.month, dt.day)
            except:
                continue
        return (0, 0, 0)
    
    def _sort_records_by_date(self, records, date_key):
        """Sort records by date in ascending order (oldest first, newest last)"""
        return sorted(records, key=lambda x: self._parse_date_for_sort(x.get(date_key, '')))
    
    def _save_price_volume_csv_to_db(self, symbol, csv_file):
        """Save price volume CSV to database - sorted by date (oldest first)"""
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                # Collect and normalize all records
                records = []
                for raw_row in reader:
                    row = self._normalize_row(raw_row)
                    trade_date = self._get_value(row, 'Date', 'DATE', 'date', 'CH_TIMESTAMP')
                    if trade_date:
                        row['_trade_date'] = trade_date
                        records.append(row)
                
                # Sort by date ascending (oldest first)
                records = self._sort_records_by_date(records, '_trade_date')
                
                # Insert sorted records
                cursor = self.db.conn.cursor()
                count = 0
                for row in records:
                    cursor.execute('''
                        INSERT OR REPLACE INTO price_volume_data 
                        (symbol, trade_date, open_price, high_price, low_price, close_price, volume, delivery_qty, delivery_pct, data_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        row['_trade_date'],
                        self._parse_float(self._get_value(row, 'Open Price', 'OPEN', 'CH_OPENING_PRICE', 'open')),
                        self._parse_float(self._get_value(row, 'High Price', 'HIGH', 'CH_TRADE_HIGH_PRICE', 'high')),
                        self._parse_float(self._get_value(row, 'Low Price', 'LOW', 'CH_TRADE_LOW_PRICE', 'low')),
                        self._parse_float(self._get_value(row, 'Close Price', 'CLOSE', 'CH_CLOSING_PRICE', 'close')),
                        self._parse_float(self._get_value(row, 'Total Traded Quantity', 'TTL_TRD_QNTY', 'CH_TOT_TRADED_QTY', 'volume')),
                        self._parse_float(self._get_value(row, 'Deliverable Qty', 'DELIV_QTY', 'COP_DELIV_QTY')),
                        self._parse_float(self._get_value(row, '% Dly Qt to Traded Qty', 'DELIV_PER', 'COP_DELIV_PERC')),
                        'nse_historical'
                    ))
                    count += 1
                self.db.conn.commit()
                print(f"    Saved {count} records from {csv_file}")
        except Exception as e:
            print(f"    Error saving {csv_file}: {e}")
    
    def _save_futures_csv_to_db(self, symbol, csv_file):
        """Save futures CSV to database - sorted by date (oldest first)"""
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                # Collect and normalize all records
                records = []
                for raw_row in reader:
                    row = self._normalize_row(raw_row)
                    trade_date = self._get_value(row, 'FH_TIMESTAMP', 'Date', 'DATE', 'date')
                    if trade_date:
                        row['_trade_date'] = trade_date
                        records.append(row)
                
                # Sort by date ascending (oldest first)
                records = self._sort_records_by_date(records, '_trade_date')
                
                # Insert sorted records
                cursor = self.db.conn.cursor()
                count = 0
                for row in records:
                    cursor.execute('''
                        INSERT OR REPLACE INTO futures_data 
                        (symbol, trade_date, expiry_date, open_price, high_price, low_price, close_price, volume, open_interest, change_in_oi)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        row['_trade_date'],
                        self._get_value(row, 'FH_EXPIRY_DT', 'Expiry', 'EXPIRY_DT', 'expiry'),
                        self._parse_float(self._get_value(row, 'FH_OPENING_PRICE', 'Open', 'OPEN')),
                        self._parse_float(self._get_value(row, 'FH_TRADE_HIGH_PRICE', 'High', 'HIGH')),
                        self._parse_float(self._get_value(row, 'FH_TRADE_LOW_PRICE', 'Low', 'LOW')),
                        self._parse_float(self._get_value(row, 'FH_CLOSING_PRICE', 'Close', 'CLOSE')),
                        self._parse_float(self._get_value(row, 'FH_TOT_TRADED_QTY', 'Volume', 'VOLUME')),
                        self._parse_float(self._get_value(row, 'FH_OPEN_INT', 'OI', 'OPEN_INT')),
                        self._parse_float(self._get_value(row, 'FH_CHANGE_IN_OI', 'Change_OI', 'CHG_IN_OI'))
                    ))
                    count += 1
                self.db.conn.commit()
                print(f"    Saved {count} futures records from {csv_file}")
        except Exception as e:
            print(f"    Error saving {csv_file}: {e}")
    
    def _save_option_chain_csv_to_db(self, symbol, csv_file):
        """Save option chain CSV to database"""
        try:
            # Extract expiry date from filename (e.g., SYMBOL_option_chain_02_Jan_2025.csv)
            import re
            expiry_match = re.search(r'option_chain_(.+)\.csv$', csv_file)
            expiry_from_file = expiry_match.group(1).replace('_', '-') if expiry_match else ''
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                cursor = self.db.conn.cursor()
                count = 0
                for row in reader:
                    expiry_date = row.get('expiryDate') or row.get('expiry_date') or expiry_from_file
                    strike_price = float(row.get('strikePrice', 0) or 0)
                    
                    if not expiry_date or not strike_price:
                        continue
                    
                    # Save CE
                    ce_oi = row.get('CE_openInterest') or row.get('CE_oi')
                    if ce_oi:
                        cursor.execute('''
                            INSERT OR REPLACE INTO option_chain_data 
                            (symbol, expiry_date, strike_price, option_type, open_interest, change_in_oi, volume, iv, ltp, bid_price, ask_price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol,
                            str(expiry_date),
                            strike_price,
                            'CE',
                            float(ce_oi or 0),
                            float(row.get('CE_changeinOpenInterest', row.get('CE_changeInOI', 0)) or 0),
                            float(row.get('CE_totalTradedVolume', row.get('CE_volume', 0)) or 0),
                            float(row.get('CE_impliedVolatility', row.get('CE_iv', 0)) or 0),
                            float(row.get('CE_lastPrice', row.get('CE_ltp', 0)) or 0),
                            float(row.get('CE_buyPrice1', row.get('CE_bidprice', row.get('CE_bidPrice', 0))) or 0),
                            float(row.get('CE_sellPrice1', row.get('CE_askPrice', 0)) or 0)
                        ))
                        count += 1
                    
                    # Save PE
                    pe_oi = row.get('PE_openInterest') or row.get('PE_oi')
                    if pe_oi:
                        cursor.execute('''
                            INSERT OR REPLACE INTO option_chain_data 
                            (symbol, expiry_date, strike_price, option_type, open_interest, change_in_oi, volume, iv, ltp, bid_price, ask_price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol,
                            str(expiry_date),
                            strike_price,
                            'PE',
                            float(pe_oi or 0),
                            float(row.get('PE_changeinOpenInterest', row.get('PE_changeInOI', 0)) or 0),
                            float(row.get('PE_totalTradedVolume', row.get('PE_volume', 0)) or 0),
                            float(row.get('PE_impliedVolatility', row.get('PE_iv', 0)) or 0),
                            float(row.get('PE_lastPrice', row.get('PE_ltp', 0)) or 0),
                            float(row.get('PE_buyPrice1', row.get('PE_bidprice', row.get('PE_bidPrice', 0))) or 0),
                            float(row.get('PE_sellPrice1', row.get('PE_askPrice', 0)) or 0)
                        ))
                        count += 1
                self.db.conn.commit()
                print(f"    Saved {count} option records from {csv_file}")
        except Exception as e:
            print(f"    Error saving {csv_file}: {e}")

    # ==================== LAYER 1 EXTRACTION ====================

    def extract_layer1_data(self, symbol):
        """
        Extract all Layer 1 (symbol-specific) data
        - JSON data saved to single JSON file
        - CSV data saved to separate CSV files
        """
        symbol = symbol.upper()
        output_dir = f"{symbol}_data"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"LAYER 1: Extracting Symbol-Specific Data for {symbol}")
        print(f"{'='*60}\n")
        
        # ===== JSON Data =====
        json_data = {}
        
        json_apis = [
            ("equity_details", lambda: self.get_equity_details(symbol)),
            ("trade_info", lambda: self.get_equity_trade_info(symbol)),
            ("corporate_info", lambda: self.get_equity_corporate_info(symbol)),
            ("intraday_data", lambda: self.get_equity_intraday_data(symbol)),
            ("historical_data", lambda: self.get_equity_historical_data(symbol)),
            ("results_comparison", lambda: self.get_results_comparison(symbol)),
        ]
        
        print("Fetching JSON data from NSE APIs...")
        for name, func in json_apis:
            print(f"  Fetching {name}...")
            try:
                text_data = func()
                # Try to parse as JSON
                try:
                    json_data[name] = json.loads(text_data)
                except:
                    json_data[name] = {"raw_text": text_data}
            except Exception as e:
                json_data[name] = {"error": str(e)}
        
        # Add IndianAPI data
        print("\nFetching data from IndianAPI...")
        try:
            stock_details = self.get_stock_details_indianapi(symbol)
            json_data["stock_details_indianapi"] = json.loads(stock_details)
        except:
            json_data["stock_details_indianapi"] = {"raw_text": stock_details}
        
        try:
            historical_stats = self.get_historical_stats_indianapi(symbol)
            json_data["historical_stats_indianapi"] = json.loads(historical_stats)
        except:
            json_data["historical_stats_indianapi"] = {"raw_text": historical_stats}
        
        # Save all JSON data to single file
        json_file = f"{output_dir}/{symbol}_layer1_data.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON data saved to: {json_file}")
        
        # Save to database if available
        if self.db:
            print("Saving to database...")
            # Save stock info
            if "equity_details" in json_data and isinstance(json_data["equity_details"], dict):
                self.db.save_stock_info(symbol, json_data["equity_details"])
            
            # Save each data type
            for data_type, data in json_data.items():
                self.db.save_layer1_json(symbol, data_type, data)
            print("  Data saved to SQLite database")
        
        # ===== CSV Data =====
        print("\n" + "-"*40)
        print("Fetching CSV data...")
        print("-"*40)
        
        # 1. Price Volume Deliverable Data (5 years, 1 year at a time)
        print("\n1. Price/Volume/Deliverable Position Data (5 years):")
        pvd_files = self.get_price_volume_deliverable_data(symbol, output_dir)
        
        # Save price volume data to database
        if self.db and pvd_files:
            print("  Saving price/volume data to database...")
            for csv_file in pvd_files:
                self._save_price_volume_csv_to_db(symbol, csv_file)
        
        # 2. Futures Price Volume Data (1 year)
        print("\n2. Futures Price/Volume Data (1 year):")
        futures_file = self.get_future_price_volume_data(symbol, output_dir)
        
        # Save futures data to database
        if self.db and futures_file:
            print("  Saving futures data to database...")
            self._save_futures_csv_to_db(symbol, futures_file)
        
        # 3. Option Chain for all expiries
        print("\n3. Option Chain Data (all expiries):")
        option_files = self.get_option_chain_all_expiries(symbol, output_dir)
        
        # Save option chain data to database
        if self.db and option_files:
            print("  Saving option chain data to database...")
            for csv_file in option_files:
                self._save_option_chain_csv_to_db(symbol, csv_file)
        
        # Summary
        print(f"\n{'='*60}")
        print("LAYER 1 EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}/")
        print(f"JSON file: {json_file}")
        print(f"CSV files created: {len(pvd_files) + (1 if futures_file else 0) + len(option_files)}")
        
        return {
            "output_dir": output_dir,
            "json_file": json_file,
            "csv_files": {
                "price_volume_deliverable": pvd_files,
                "futures": futures_file,
                "option_chain": option_files
            }
        }

    # ==================== LAYER 3 EXTRACTION ====================
    
    def extract_layer3_data(self, symbol, fii_dii_db_path="fii_dii_trade.db"):
        """
        Extract Layer 3 (Index Data - NIFTY 50)
        Includes: Market status, Index intraday, Index history, VIX, Bulk/Block deals, Index futures, Index option chain
        """
        symbol = symbol.upper()
        output_dir = f"{symbol}_data"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"LAYER 3: Extracting Index Data (NIFTY 50)")
        print(f"{'='*60}")
        if self.db:
            print(f"Database: {self.db.db_path}")
        else:
            print("WARNING: No database connection - Layer 3 data will NOT be saved to DB!")
        print()
        
        layer3_data = {
            "index": "NIFTY 50",
            "extracted_at": datetime.now().isoformat()
        }
        
        # 1. Market Status
        print("1. Fetching Market Status...")
        try:
            market_status = self.get_market_status()
            layer3_data["market_status"] = json.loads(market_status)
        except:
            layer3_data["market_status"] = {"raw": market_status}
        
        # 2. Index Intraday Data
        print("2. Fetching Index Intraday Data...")
        try:
            intraday = self.get_index_intraday_data("NIFTY 50")
            layer3_data["index_intraday"] = json.loads(intraday)
        except:
            layer3_data["index_intraday"] = {"raw": intraday}
        
        # 3. Index Option Chain Contract Info
        print("3. Fetching Index Option Chain Contract Info...")
        try:
            contract_info = self.get_index_option_chain_contract_info("NIFTY")
            layer3_data["index_option_chain_contract_info"] = json.loads(contract_info)
        except:
            layer3_data["index_option_chain_contract_info"] = {"raw": contract_info}
        
        # 4. Index History Data (1 year) - Save to CSV and DB
        print("\n4. Fetching Index History Data (1 year)...")
        try:
            history_raw = self.get_index_history_data("NIFTY 50")
            history_json = json.loads(history_raw)
            # Handle different response formats
            if isinstance(history_json, list):
                records = history_json
            elif isinstance(history_json, dict):
                records = history_json.get("data", history_json.get("records", []))
            else:
                records = []
            print(f"    Found {len(records)} index history records")
            
            if records:
                # Sort by date ascending (oldest first)
                sorted_records = sorted(records, key=lambda x: self._parse_date_for_sort(
                    x.get("EOD_TIMESTAMP", x.get("Date", ""))
                ))
                
                filename = f"{output_dir}/NIFTY50_index_history.csv"
                fieldnames = sorted_records[0].keys()
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sorted_records)
                print(f"    Saved: {filename}")
                
                # Save to database (already sorted)
                if self.db:
                    cursor = self.db.conn.cursor()
                    for rec in sorted_records:
                        cursor.execute('''
                            INSERT OR REPLACE INTO index_history 
                            (index_name, trade_date, open_value, high_value, low_value, close_value, volume, turnover)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            "NIFTY 50",
                            rec.get("EOD_TIMESTAMP", rec.get("Date", "")),
                            float(str(rec.get("EOD_OPEN_INDEX_VAL", 0)).replace(",", "") or 0),
                            float(str(rec.get("EOD_HIGH_INDEX_VAL", 0)).replace(",", "") or 0),
                            float(str(rec.get("EOD_LOW_INDEX_VAL", 0)).replace(",", "") or 0),
                            float(str(rec.get("EOD_CLOSE_INDEX_VAL", 0)).replace(",", "") or 0),
                            float(str(rec.get("HIT_TRADED_QTY", rec.get("EOD_INDEX_TRADING_VOL", 0))).replace(",", "") or 0),
                            float(str(rec.get("HIT_TURN_OVER", rec.get("EOD_INDEX_TRADING_TURNOVER", 0))).replace(",", "") or 0)
                        ))
                    self.db.conn.commit()
                    print(f"    Saved {len(sorted_records)} records to database")
        except Exception as e:
            print(f"    Error: {e}")
        
        # 5. India VIX Data (1 year) - Returns JSON, save to CSV and DB
        print("\n5. Fetching India VIX Data (1 year)...")
        try:
            vix_raw = self.get_india_vix_data()
            
            # VIX API returns JSON
            vix_json = json.loads(vix_raw)
            # Handle different response formats
            if isinstance(vix_json, list):
                records = vix_json
            elif isinstance(vix_json, dict):
                records = vix_json.get("data", vix_json.get("records", []))
            else:
                records = []
            print(f"    Found {len(records)} VIX records")
            
            if records:
                # Sort by date ascending (oldest first)
                sorted_records = sorted(records, key=lambda x: self._parse_date_for_sort(
                    x.get('EOD_TIMESTAMP', '')
                ))
                
                # Save as CSV
                filename = f"{output_dir}/india_vix_history.csv"
                fieldnames = sorted_records[0].keys()
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sorted_records)
                print(f"    Saved: {filename}")
                
                # Save to database (already sorted)
                if self.db:
                    cursor = self.db.conn.cursor()
                    count = 0
                    for rec in sorted_records:
                        trade_date = rec.get('EOD_TIMESTAMP', '')
                        if not trade_date:
                            continue
                        cursor.execute('''
                            INSERT OR REPLACE INTO vix_history 
                            (trade_date, open_value, high_value, low_value, close_value, prev_close, change_value, change_pct)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            trade_date,
                            float(rec.get('EOD_OPEN_INDEX_VAL', 0) or 0),
                            float(rec.get('EOD_HIGH_INDEX_VAL', 0) or 0),
                            float(rec.get('EOD_LOW_INDEX_VAL', 0) or 0),
                            float(rec.get('EOD_CLOSE_INDEX_VAL', 0) or 0),
                            float(rec.get('EOD_PREV_CLOSE', 0) or 0),
                            float(rec.get('VIX_PTS_CHG', 0) or 0),
                            float(rec.get('VIX_PERC_CHG', 0) or 0)
                        ))
                        count += 1
                    self.db.conn.commit()
                    print(f"    Saved {count} VIX records to database")
            else:
                # Save raw response
                filename = f"{output_dir}/india_vix_history.json"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(vix_raw)
                print(f"    Saved raw: {filename}")
        except Exception as e:
            print(f"    Error: {e}")
        
        # 6. Bulk Deals Data (1 year) - Save CSV, filter symbol for DB
        print("\n6. Fetching Bulk Deals Data (1 year)...")
        try:
            bulk_data = self.get_bulk_deals_data()
            filename = f"{output_dir}/bulk_deals_all.csv"
            
            # Parse and save with clean headers
            import io
            clean_data = bulk_data.replace('\ufeff', '').replace('Ã¯Â»Â¿', '')
            reader = csv.DictReader(io.StringIO(clean_data))
            clean_rows = []
            for raw_row in reader:
                row = self._normalize_row(raw_row)
                clean_rows.append({
                    'Date': self._get_value(row, 'Date', 'DATE', 'date'),
                    'Symbol': self._get_value(row, 'Symbol', 'SYMBOL', 'symbol'),
                    'Security Name': self._get_value(row, 'Security Name', 'SecurityName'),
                    'Client Name': self._get_value(row, 'Client Name', 'CLIENT_NAME'),
                    'Buy / Sell': self._get_value(row, 'Buy / Sell', 'Buy/Sell', 'BUY_SELL'),
                    'Quantity Traded': self._get_value(row, 'Quantity Traded', 'Quantity'),
                    'Trade Price': self._get_value(row, 'Trade Price / Wght. Avg. Price', 'Trade Price'),
                    'Remarks': self._get_value(row, 'Remarks', 'REMARKS')
                })
            
            if clean_rows:
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=clean_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(clean_rows)
            print(f"    Saved: {filename}")
            
            # Filter and save symbol-specific data to database
            if self.db:
                import io
                # Remove BOM if present (various encodings)
                clean_data = bulk_data.replace('\ufeff', '').replace('Ã¯Â»Â¿', '')
                reader = csv.DictReader(io.StringIO(clean_data))
                cursor = self.db.conn.cursor()
                count = 0
                for raw_row in reader:
                    row = self._normalize_row(raw_row)
                    row_symbol = self._get_value(row, 'Symbol', 'SYMBOL', 'symbol')
                    deal_date = self._get_value(row, 'Date', 'DATE', 'date', 'Deal Date')
                    if row_symbol.upper() == symbol and deal_date:
                        cursor.execute('''
                            INSERT OR REPLACE INTO bulk_deals 
                            (symbol, deal_date, client_name, deal_type, quantity, price)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol,
                            deal_date,
                            self._get_value(row, 'Client Name', 'CLIENT_NAME', 'ClientName'),
                            self._get_value(row, 'Buy / Sell', 'Buy/Sell', 'BUY_SELL', 'Buy/Sell'),
                            self._parse_float(self._get_value(row, 'Quantity Traded', 'Quantity', 'QTY', 'QuantityTraded')),
                            self._parse_float(self._get_value(row, 'Trade Price / Wght. Avg. Price', 'Trade Price', 'PRICE', 'TradePrice'))
                        ))
                        count += 1
                self.db.conn.commit()
                print(f"    Saved {count} bulk deals for {symbol} to database")
        except Exception as e:
            print(f"    Error: {e}")
        
        # 7. Block Deals Data (1 year) - Save CSV, filter symbol for DB
        print("\n7. Fetching Block Deals Data (1 year)...")
        try:
            block_data = self.get_block_deals_data()
            filename = f"{output_dir}/block_deals_all.csv"
            
            # Parse and save with clean headers
            import io
            clean_data = block_data.replace('\ufeff', '').replace('Ã¯Â»Â¿', '')
            reader = csv.DictReader(io.StringIO(clean_data))
            clean_rows = []
            for raw_row in reader:
                row = self._normalize_row(raw_row)
                clean_rows.append({
                    'Date': self._get_value(row, 'Date', 'DATE', 'date'),
                    'Symbol': self._get_value(row, 'Symbol', 'SYMBOL', 'symbol'),
                    'Security Name': self._get_value(row, 'Security Name', 'SecurityName'),
                    'Client Name': self._get_value(row, 'Client Name', 'CLIENT_NAME'),
                    'Buy / Sell': self._get_value(row, 'Buy / Sell', 'Buy/Sell', 'BUY_SELL'),
                    'Quantity Traded': self._get_value(row, 'Quantity Traded', 'Quantity'),
                    'Trade Price': self._get_value(row, 'Trade Price / Wght. Avg. Price', 'Trade Price'),
                    'Remarks': self._get_value(row, 'Remarks', 'REMARKS')
                })
            
            if clean_rows:
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=clean_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(clean_rows)
            print(f"    Saved: {filename}")
            
            # Filter and save symbol-specific data to database
            if self.db:
                import io
                # Remove BOM if present (various encodings)
                clean_data = block_data.replace('\ufeff', '').replace('Ã¯Â»Â¿', '')
                reader = csv.DictReader(io.StringIO(clean_data))
                cursor = self.db.conn.cursor()
                count = 0
                for raw_row in reader:
                    row = self._normalize_row(raw_row)
                    row_symbol = self._get_value(row, 'Symbol', 'SYMBOL', 'symbol')
                    deal_date = self._get_value(row, 'Date', 'DATE', 'date', 'Deal Date')
                    if row_symbol.upper() == symbol and deal_date:
                        cursor.execute('''
                            INSERT OR REPLACE INTO block_deals 
                            (symbol, deal_date, client_name, deal_type, quantity, price)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol,
                            deal_date,
                            self._get_value(row, 'Client Name', 'CLIENT_NAME', 'ClientName'),
                            self._get_value(row, 'Buy / Sell', 'Buy/Sell', 'BUY_SELL', 'Buy/Sell'),
                            self._parse_float(self._get_value(row, 'Quantity Traded', 'Quantity', 'QTY', 'QuantityTraded')),
                            self._parse_float(self._get_value(row, 'Trade Price / Wght. Avg. Price', 'Trade Price', 'PRICE', 'TradePrice'))
                        ))
                        count += 1
                self.db.conn.commit()
                print(f"    Saved {count} block deals for {symbol} to database")
        except Exception as e:
            print(f"    Error: {e}")
        
        # 8. Index Futures Data (3 months) - Returns JSON, save to CSV and DB
        print("\n8. Fetching Index Futures Data (3 months)...")
        try:
            futures_raw = self.get_index_future_price_volume_data("NIFTY")
            
            # API returns JSON
            futures_json = json.loads(futures_raw)
            # Handle different response formats
            if isinstance(futures_json, list):
                records = futures_json
            elif isinstance(futures_json, dict):
                records = futures_json.get("data", futures_json.get("records", []))
            else:
                records = []
            print(f"    Found {len(records)} index futures records")
            
            if records:
                # Sort by date ascending (oldest first)
                sorted_records = sorted(records, key=lambda x: self._parse_date_for_sort(
                    x.get('FH_TIMESTAMP', '')
                ))
                
                # Save as CSV
                filename = f"{output_dir}/NIFTY_index_futures.csv"
                fieldnames = sorted_records[0].keys()
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sorted_records)
                print(f"    Saved: {filename}")
                
                # Save to database (already sorted)
                if self.db:
                    cursor = self.db.conn.cursor()
                    count = 0
                    for rec in sorted_records:
                        trade_date = rec.get('FH_TIMESTAMP', '')
                        if not trade_date:
                            continue
                        cursor.execute('''
                            INSERT OR REPLACE INTO index_futures_data 
                            (index_name, trade_date, expiry_date, open_price, high_price, low_price, close_price, volume, open_interest, change_in_oi)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            "NIFTY",
                            trade_date,
                            rec.get('FH_EXPIRY_DT', ''),
                            float(rec.get('FH_OPENING_PRICE', 0) or 0),
                            float(rec.get('FH_TRADE_HIGH_PRICE', 0) or 0),
                            float(rec.get('FH_TRADE_LOW_PRICE', 0) or 0),
                            float(rec.get('FH_CLOSING_PRICE', 0) or 0),
                            float(rec.get('FH_TOT_TRADED_QTY', 0) or 0),
                            float(rec.get('FH_OPEN_INT', 0) or 0),
                            float(rec.get('FH_CHANGE_IN_OI', 0) or 0)
                        ))
                        count += 1
                    self.db.conn.commit()
                    print(f"    Saved {count} index futures records to database")
            else:
                # Save raw response
                filename = f"{output_dir}/NIFTY_index_futures.json"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(futures_raw)
                print(f"    Saved raw: {filename}")
        except Exception as e:
            print(f"    Error: {e}")
        
        # 9. Index Option Chain (all expiries) - Save CSV and DB
        print("\n9. Fetching Index Option Chain (all expiries)...")
        try:
            option_files, option_records = self.get_nse_index_option_chain("NIFTY", output_dir)
            
            # Save to database
            if self.db and option_records:
                cursor = self.db.conn.cursor()
                count = 0
                for row in option_records:
                    expiry = row.get('expiryDate', '')
                    strike = float(row.get('strikePrice', 0) or 0)
                    
                    # Save CE
                    if row.get('CE_openInterest'):
                        cursor.execute('''
                            INSERT OR REPLACE INTO index_option_chain 
                            (index_name, expiry_date, strike_price, option_type, open_interest, change_in_oi, volume, iv, ltp, bid_price, ask_price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            "NIFTY",
                            expiry,
                            strike,
                            'CE',
                            float(row.get('CE_openInterest', 0) or 0),
                            float(row.get('CE_changeinOpenInterest', 0) or 0),
                            float(row.get('CE_totalTradedVolume', 0) or 0),
                            float(row.get('CE_impliedVolatility', 0) or 0),
                            float(row.get('CE_lastPrice', 0) or 0),
                            float(row.get('CE_buyPrice1', row.get('CE_bidprice', 0)) or 0),
                            float(row.get('CE_sellPrice1', row.get('CE_askPrice', 0)) or 0)
                        ))
                        count += 1
                    
                    # Save PE
                    if row.get('PE_openInterest'):
                        cursor.execute('''
                            INSERT OR REPLACE INTO index_option_chain 
                            (index_name, expiry_date, strike_price, option_type, open_interest, change_in_oi, volume, iv, ltp, bid_price, ask_price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            "NIFTY",
                            expiry,
                            strike,
                            'PE',
                            float(row.get('PE_openInterest', 0) or 0),
                            float(row.get('PE_changeinOpenInterest', 0) or 0),
                            float(row.get('PE_totalTradedVolume', 0) or 0),
                            float(row.get('PE_impliedVolatility', 0) or 0),
                            float(row.get('PE_lastPrice', 0) or 0),
                            float(row.get('PE_buyPrice1', row.get('PE_bidprice', 0)) or 0),
                            float(row.get('PE_sellPrice1', row.get('PE_askPrice', 0)) or 0)
                        ))
                        count += 1
                self.db.conn.commit()
                print(f"    Saved {count} index option records to database")
        except Exception as e:
            print(f"    Error: {e}")
        
        # 10. Get FII/DII data from separate database
        print("\n10. Fetching FII/DII Trade Data...")
        try:
            # First update FII/DII database
            fii_dii_db = FiiDiiDatabase(fii_dii_db_path)
            fii_dii_raw = self.get_fii_dii_trade_react()
            fii_dii_json = json.loads(fii_dii_raw)
            
            # Save to database and get count
            saved_count = fii_dii_db.save_fii_dii_data(fii_dii_json)
            print(f"    Saved {saved_count} FII/DII records to {fii_dii_db_path}")
            
            # Get latest data for layer3 json (include raw API response too)
            layer3_data["fii_dii_trade"] = {
                "latest_from_api": fii_dii_json,
                "historical_from_db": fii_dii_db.get_latest_data()
            }
            fii_dii_db.close()
        except Exception as e:
            print(f"    Error: {e}")
            layer3_data["fii_dii_trade"] = {"error": str(e)}
        
        # Save Layer 3 JSON
        json_file = f"{output_dir}/{symbol}_layer3_data.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(layer3_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON data saved to: {json_file}")
        
        # Save Layer 3 JSON to database
        if self.db:
            print("Saving Layer 3 JSON to database...")
            for data_type, data in layer3_data.items():
                if data_type not in ["symbol", "extracted_at", "index"]:  # Skip metadata
                    self.db.save_layer3_json(symbol, data_type, data)
            print("  Layer 3 JSON saved to database")
        
        # Summary
        print(f"\n{'='*60}")
        print("LAYER 3 EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"JSON file: {json_file}")
        print(f"FII/DII database: {fii_dii_db_path}")
        
        return {
            "output_dir": output_dir,
            "json_file": json_file,
            "fii_dii_db": fii_dii_db_path
        }
    
    def generate_trading_signals(self, symbol):
        """
        Generate comprehensive trading signals from extracted database data.
        Creates {symbol}_compiled.json file with all signals.
        """
        if not self.db:
            print("WARNING: No database connection - cannot generate signals!")
            return None
        
        symbol = symbol.upper()
        output_dir = f"{symbol}_data"
        
        print(f"\n{'='*60}")
        print(f"GENERATING TRADING SIGNALS FOR {symbol}")
        print(f"{'='*60}")
        
        try:
            signal_gen = SignalGenerator(self.db.db_path)
            output_file = signal_gen.save_compiled_json(symbol, output_dir)
            signal_gen.close()
            
            print(f"\n{'='*60}")
            print("SIGNAL GENERATION COMPLETE")
            print(f"{'='*60}")
            print(f"Output file: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return None


def main():
    ticker = input("Enter stock ticker symbol: ").strip()
    if not ticker:
        print("No ticker provided!")
        return
    
    # Create output directory and database
    output_dir = f"{ticker.upper()}_data"
    os.makedirs(output_dir, exist_ok=True)
    db_path = f"{output_dir}/{ticker.upper()}_stock_data.db"
    fii_dii_db_path = "fii_dii_trade.db"  # Separate database for FII/DII data
    
    # Initialize with database
    nse = NseIndia(db_path=db_path)
    
    print(f"\nStock Database: {db_path}")
    print(f"FII/DII Database: {fii_dii_db_path}")
    
    # Extract Layer 1 data (Symbol-specific)
    layer1_result = nse.extract_layer1_data(ticker)
    
    # Extract Layer 2 data (Sectorial)
    layer2_result = nse.extract_layer2_data(ticker, layer1_result["json_file"])
    
    # Extract Layer 3 data (Index - NIFTY 50)
    layer3_result = nse.extract_layer3_data(ticker, fii_dii_db_path)
    
    # Generate Trading Signals from database
    print("\n" + "="*60)
    print("GENERATING TRADING SIGNALS")
    print("="*60)
    signal_file = nse.generate_trading_signals(ticker)
    
    # Close database connection
    if nse.db:
        nse.db.close()
    
    print(f"\n{'='*60}")
    print("ALL OPERATIONS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {layer1_result['output_dir']}/")
    print(f"Stock Database: {db_path}")
    print(f"FII/DII Database: {fii_dii_db_path}")
    print(f"Layer 1 JSON: {layer1_result['json_file']}")
    if layer2_result:
        print(f"Layer 2 JSON: {layer2_result['json_file']}")
        print(f"Matched sectors: {len(layer2_result['matched_sectors'])}")
    if layer3_result:
        print(f"Layer 3 JSON: {layer3_result['json_file']}")
    if signal_file:
        print(f"Trading Signals: {signal_file}")


if __name__ == "__main__":
    main()
