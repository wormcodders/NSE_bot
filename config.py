"""
Configuration for Stock Signal Bot

Edit this file to configure your bot.
"""

# Telegram Bot Token (Get from @BotFather on Telegram)
TELEGRAM_BOT_TOKEN = "8091329681:AAERVD30ntt3KXagF48a1MoY0-aQG5ERhC4"

# Admin User IDs (These users can use the /test command)
# Find your user ID by messaging @userinfobot on Telegram
ADMIN_IDS = [
    1442529414,  # Replace with your actual Telegram user ID
    # Add more admin IDs as needed
]

# Scheduled Analysis Times (24-hour format)
# Market typically opens at 9:30 AM and closes at 3:30 PM (IST)
SCHEDULED_TIMES = [
    "9:30",   # Pre-market analysis
    "12:00",  # Mid-day analysis
    "15:00",  # Pre-close analysis
]

# Analysis Configuration
# Number of days of historical data to fetch for analysis
ANALYSIS_PERIOD_DAYS = 90

# RSI Configuration
RSI_WINDOW = 14  # Period for RSI calculation
RSI_BUY_THRESHOLD = 40  # Minimum RSI for buy signal (0-100)
RSI_SELL_THRESHOLD = 60  # Minimum RSI for sell signal (0-100)

# Lookback period for detecting crossovers
RECENT_DAYS_LOOKBACK = 5

# Telegram Message Settings
# Rate limiting delay between messages (seconds)
TELEGRAM_RATE_LIMIT_DELAY = 0.05

# Enable/disable features
ENABLE_SCHEDULER = True
ENABLE_WEBHOOK = False

# Webhook settings (if ENABLE_WEBHOOK is True)
WEBHOOK_HOST = "0.0.0.0"
WEBHOOK_PORT = 8443
WEBHOOK_URL_PATH = "/webhook"
