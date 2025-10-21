#!/usr/bin/env python3
"""
Direct Python script to run the GSE Sentiment Analysis Dashboard
This avoids virtual environment path issues
"""

import subprocess
import sys
import os

def run_dashboard():
    """Run the Streamlit dashboard directly"""
    print("="*60)
    print("GSE SENTIMENT ANALYSIS DASHBOARD")
    print("="*60)
    print("Starting dashboard...")

    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(current_dir, 'working_dashboard.py')

        # Check if dashboard file exists
        if not os.path.exists(dashboard_path):
            print(f"❌ Error: Dashboard file not found at {dashboard_path}")
            return

        print(f"📁 Dashboard file: {dashboard_path}")
        print("🌐 Opening dashboard in browser...")
        print("📸 Take screenshots of these 6 figures for your thesis:")
        print("   • Figure 4.1: Data sources distribution (Executive Summary tab)")
        print("   • Figure 4.2: Sentiment by source (Data Sources tab)")
        print("   • Figure 4.3: Model performance (Model Performance tab)")
        print("   • Figure 4.4: Correlation matrix (Correlation Studies tab)")
        print("   • Figure 4.5: Confidence levels (Real-Time Predictions tab)")
        print("   • Figure 4.6: Sector analysis (Time Series Analysis tab)")
        print()
        print("💡 Tip: Use Ctrl+C to stop the dashboard when done")
        print("="*60)

        # Run streamlit directly with python -m
        cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path]
        subprocess.run(cmd, cwd=current_dir)

    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the correct directory")
        print("2. Try: python -m streamlit run working_dashboard.py")
        print("3. Or: streamlit run working_dashboard.py")

if __name__ == "__main__":
    run_dashboard()