"""
Logging Configuration Module

Centralizes logging for the application.
Outputs to console and daily log file (app_YYYYMMDD.log).
"""

# Core dependencies
import logging
from datetime import datetime

# Suppress Streamlit framework warnings
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.CRITICAL)
logging.getLogger('streamlit').setLevel(logging.CRITICAL)

# Initialize logger
logger = logging.getLogger("loan_app")
logger.setLevel(logging.INFO)

# Console handler for terminal output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler for daily logging
log_filename = f"app_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

# Set log format with timestamp
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Prevent duplicate logs
logger.propagate = False