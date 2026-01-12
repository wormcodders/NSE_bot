# Stock Signal Bot

Automated stock buy/sell signal scanner with Telegram notifications. This bot analyzes stock tickers using technical indicators (MACD and RSI) and sends buy/sell signals to subscribed users at scheduled times.

## Features

- **Automated Analysis**: Runs at 9:30 AM, 12:00 PM, and 3:00 PM every trading day
- **Technical Indicators**: Uses MACD crossovers and RSI for signal generation
- **Telegram Notifications**: Sends signals directly to subscribed users via Telegram bot
- **User Management**: Simple subscribe/unsubscribe commands
- **On-Demand Signals**: Users can request current signals anytime with `/signals`
- **Test Mode**: Admin-only `/test` command for verification without notifying subscribers

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Telegram account
- Telegram bot token (from @BotFather)

### Installation

1. Clone or download this repository:

```bash
git clone <repository-url>
cd stock_signal_bot
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure the bot:

Edit `config.py` and set your Telegram bot token and admin user IDs:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
ADMIN_IDS = [123456789, 987654321]
```

5. Create the tickers file:

Create an Excel file named `tickers.xlsx` with stock ticker symbols in the first column. You can use any Excel-compatible format (.xlsx, .xls).

6. Run the bot:

```bash
python bot.py
```

## File Structure

```
stock_signal_bot/
├── bot.py              # Main application script
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── tickers.xlsx        # Stock ticker list (create this)
├── subscribers.json    # User data (auto-generated)
├── bot.log            # Application log (auto-generated)
└── README.md          # This file
```

## Configuration

### Telegram Bot Setup

1. Open Telegram and search for @BotFather
2. Send `/newbot` to create a new bot
3. Follow the instructions to set a name and username
4. Copy the API token provided
5. Add the token to `config.py`

### Finding Your Telegram User ID

1. Search for @userinfobot on Telegram
2. Send `/start` to the bot
3. Copy your user ID
4. Add it to the `ADMIN_IDS` list in `config.py`

### Admin-Only Test Button

The `/test` command allows administrators to verify the entire application flow:

```
/test
```

**What it does:**
- Runs the analysis immediately (no waiting for scheduled times)
- Sends test signals only to you (the admin who ran the test)
- Does NOT notify regular subscribers
- Clearly marks messages as "TEST MODE"

**Requirements:**
- Only users in `ADMIN_IDS` can use this command
- Other users will receive an "Unauthorized" message

**Expected output:**
```
🧪 TEST MODE 🧪
━━━━━━━━━━━━━━━━
⚠️ This is a test message - NOT a live signal

📊 Scan Time: 2025-12-28 15:30:00
📈 Tickers Analyzed: 50
❌ Errors: 0
⏱️ Execution Time: 2.45s

🟢 BUY SIGNALS
━━━━━━━━━━━━━━━━
• AAPL - $150.00
  RSI: 45.2

🔴 SELL SIGNALS
━━━━━━━━━━━━━━━━
• TSLA - $200.00
  RSI: 68.5

━━━━━━━━━━━━━━━━
✅ Test completed successfully
📱 Real subscribers were NOT notified
```

## User Commands

### For End Users

| Command | Description |
|---------|-------------|
| `/start` | Subscribe to signal notifications |
| `/stop` | Unsubscribe from notifications |
| `/signals` | Get current buy/sell signals on demand |
| `/status` | Check subscription status |
| `/help` | Show all available commands |

### For Admins

| Command | Description |
|---------|-------------|
| `/test` | Run test analysis (sends signals to you only) |

## Signal Logic

### Buy Signal

A buy signal is generated when:
- MACD line crosses ABOVE the Signal line
- RSI is at least 40 (not oversold)

### Sell Signal

A sell signal is generated when:
- MACD line crosses BELOW the Signal line

### Technical Indicators

- **MACD (Moving Average Convergence Divergence)**: Identifies momentum changes
- **RSI (Relative Strength Index)**: Measures overbought/oversold conditions

## Scheduled Analysis

The bot automatically runs analysis at these times:
- **9:30 AM**: Pre-market signal check
- **12:00 PM**: Mid-day signal update
- **3:00 PM**: Pre-close signal summary

All active subscribers receive the signals after each analysis run.

## Maintenance

### Updating Tickers

After each market expiry, update `tickers.xlsx` with the new list of stock symbols. The bot will automatically pick up the changes on its next analysis run.

### Checking Logs

Application logs are written to `bot.log`. Check this file for:
- Analysis results
- Notification delivery status
- Error messages

```bash
tail -f bot.log
```

### Viewing Subscribers

The `subscribers.json` file contains all subscriber data. Each entry includes:
- User ID
- Name
- Username
- Subscription status
- Subscription timestamp

## Troubleshooting

### Bot Not Responding

1. Check if the script is running:
```bash
ps aux | grep bot.py
```

2. Check the log file for errors:
```bash
cat bot.log | tail -50
```

3. Restart the bot:
```bash
pkill -f bot.py
python bot.py &
```

### No Signals Generated

1. Verify `tickers.xlsx` exists and contains valid ticker symbols
2. Check that tickers are in the correct format (uppercase, no spaces)
3. Verify internet connectivity for yfinance API access
4. Check log for specific ticker errors

### Telegram Messages Not Delivered

1. Verify the bot token is correct in `config.py`
2. Check that users have sent `/start` to subscribe
3. Look for rate limiting or API errors in `bot.log`
4. Ensure the bot has permission to send messages

### Test Command Not Working

1. Verify your user ID is in `ADMIN_IDS`
2. Make sure you restarted the bot after adding your ID
3. Check the log file for authorization errors

## Customization

### Changing Analysis Parameters

Edit `config.py` to modify:

- `ANALYSIS_PERIOD_DAYS`: Historical data range (default: 180 days)
- `RSI_WINDOW`: RSI calculation period (default: 14)
- `RSI_BUY_THRESHOLD`: Minimum RSI for buy signals (default: 40)
- `SCHEDULED_TIMES`: Analysis schedule

### Changing Signal Thresholds

Edit `bot.py` to modify signal generation logic:

```python
# In the analyze_ticker function
RSI_BUY_THRESHOLD = 50  # Higher = more conservative
```

## Security Notes

- Keep your bot token confidential
- Only add trusted user IDs to `ADMIN_IDS`
- The bot does not store passwords or sensitive data
- Subscriber data is stored in plain JSON (consider encryption for production)

## License

This project is for educational purposes only. Stock signals are not financial advice. Always do your own research before making investment decisions.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the application logs
3. Ensure all configuration is correct
