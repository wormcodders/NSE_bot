"""
Integration Patch for bot.py

To enable web dashboard features, add the following modifications to your existing bot.py file.

Location: Around line 734-804 in the scheduled_analysis function
"""

INTEGRATION_PATCH = '''
# ===== ADD THIS IMPORT AT THE TOP OF bot.py (around line 54) =====

# Import web integration module
try:
    from bot_web import save_signals_for_web
    WEB_INTEGRATION = True
except ImportError:
    WEB_INTEGRATION = False
    print("⚠️ Web integration not available. Run web_app.py separately for dashboard.")


# ===== MODIFY scheduled_analysis FUNCTION (around line 734) =====

async def scheduled_analysis(application: Application) -> None:
    """Run scheduled analysis and send notifications to all subscribers."""
    logger.info("Starting scheduled analysis")
    
    # Get current time for scheduling record
    current_time = datetime.now()
    scheduled_time = current_time.strftime("%H:%M")
    
    # Run market-wide analysis
    buy_signals, sell_signals, summary = run_analysis()
    market_message = format_signal_message(buy_signals, sell_signals, summary, is_test=False)
    
    # ===== ADD THIS: Save signals for web dashboard =====
    if WEB_INTEGRATION:
        try:
            save_signals_for_web(buy_signals, sell_signals, summary, scheduled_time)
            logger.info("Signals saved for web dashboard")
        except Exception as e:
            logger.error(f"Failed to save signals for web: {e}")
    # ====================================================
    
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


# ===== ALSO UPDATE run_test FUNCTION (around line 847) =====

async def run_test(application: Application, user_id: int) -> str:
    """Run a test analysis and send results only to the admin."""
    global test_mode
    test_mode = True
    
    try:
        await application.bot.send_message(
            chat_id=user_id,
            text="🧪 *Test Mode Activated*\\n\\n⏳ Analyzing tickers...",
        )
        
        buy_signals, sell_signals, summary = run_analysis()
        
        # ===== ADD THIS: Save test signals for web =====
        if WEB_INTEGRATION:
            try:
                save_signals_for_web(buy_signals, sell_signals, summary, "TEST")
                logger.info("Test signals saved for web dashboard")
            except Exception as e:
                logger.error(f"Failed to save test signals: {e}")
        # ==============================================
        
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
'''

print("""
================================================================================
                    WEB APPLICATION INTEGRATION GUIDE
================================================================================

Your web application has been created successfully! Here's how to use it:

📁 FILES CREATED:
   - bot_web.py       : Signal storage module for web dashboard
   - web_app.py       : Flask web application
   - templates/index.html : Dashboard HTML template
   - static/style.css : Dashboard styling
   - static/app.js    : Dashboard JavaScript

🚀 TO RUN THE WEB DASHBOARD:

   1. Install required packages:
      pip install flask flask-cors

   2. Start the web server:
      python web_app.py

   3. Open your browser to:
      http://localhost:5000

🔗 TO INTEGRATE WITH YOUR EXISTING bot.py:

   1. Copy bot_web.py to your bot.py directory
   2. Add the import to the top of bot.py:
      
      try:
          from bot_web import save_signals_for_web
          WEB_INTEGRATION = True
      except ImportError:
          WEB_INTEGRATION = False
          print("Web integration not available")

   3. In the scheduled_analysis function (around line 740),
      add this after run_analysis():
      
      if WEB_INTEGRATION:
          save_signals_for_web(buy_signals, sell_signals, summary, scheduled_time)

📊 FEATURES:
   ✓ Live Signals Tab - Shows buy/sell signals from scheduled scans
   ✓ Auto-refresh - Updates every 30 seconds
   ✓ Depth Analysis Tab - Search for comprehensive AI-powered analysis
   ✓ Dark Theme - Professional financial terminal aesthetic
   ✓ Mobile Responsive - Works on all devices

⚙️ SCHEDULED SCAN TIMES (from config.py):
   - 9:30 AM  : Pre-market analysis
   - 12:00 PM : Mid-day analysis
   - 3:00 PM  : Pre-close analysis

📝 NOTE: The web dashboard shows ONLY signals (buy/sell), 
   NOT portfolio information as per your requirements.

================================================================================
""")
