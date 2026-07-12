"""
Shared pytest configuration. Adds the scripts directory to sys.path so
individual test files can import script modules directly.
"""
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))
