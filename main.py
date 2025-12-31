#!/usr/bin/env python3
"""
Color Cohesion Analyzer
=======================

A professional-grade tool for analyzing color palettes and evaluating
visual coherence across images and video sequences.

Designed for filmmakers and visual artists working with cinematic imagery.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui import run_app

if __name__ == "__main__":
    run_app()
