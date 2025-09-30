#!/usr/bin/env python3
"""
GSE Sentiment Analysis System - Screenshot Capture Guide
Helps users capture dashboard screenshots for research documentation
"""

import os
import time
import webbrowser
from pathlib import Path

def print_screenshot_guide():
    """Print comprehensive screenshot capture guide"""

    print("=" * 70)
    print("GSE SENTIMENT ANALYSIS - SCREENSHOT CAPTURE GUIDE")
    print("=" * 70)

    print("\nWHY CAPTURE SCREENSHOTS?")
    print("-" * 40)
    print("• Document system capabilities for research")
    print("• Include in dissertation appendices")
    print("• Demonstrate to stakeholders and investors")
    print("• Support academic publications")
    print("• Validate system functionality")

    print("\nRECOMMENDED SCREENSHOTS TO CAPTURE")
    print("-" * 40)

    screenshots = [
        {
            "tab": "Executive Summary",
            "filename": "research_figure_1_executive_summary.png",
            "purpose": "System overview and key metrics",
            "key_elements": ["Performance metrics", "Market sentiment overview", "System status"]
        },
        {
            "tab": "Model Performance Analysis",
            "filename": "research_figure_2_model_performance.png",
            "purpose": "Algorithm comparison and validation",
            "key_elements": ["Accuracy comparison", "Statistical significance", "Model rankings"]
        },
        {
            "tab": "Sentiment-Time Series Analysis",
            "filename": "research_figure_3_sentiment_trends.png",
            "purpose": "Trend visualization and volatility",
            "key_elements": ["Time series charts", "Moving averages", "Volatility indicators"]
        },
        {
            "tab": "Correlation Studies",
            "filename": "research_figure_4_correlation_analysis.png",
            "purpose": "Granger causality and relationships",
            "key_elements": ["Correlation matrices", "Causality test results", "Statistical significance"]
        },
        {
            "tab": "Real-Time Predictions",
            "filename": "research_figure_5_predictions.png",
            "purpose": "Live forecasting demonstration",
            "key_elements": ["Prediction results", "Model comparison", "Confidence levels"]
        },
        {
            "tab": "Manual Sentiment Input",
            "filename": "research_figure_6_manual_input.png",
            "purpose": "Hybrid intelligence system",
            "key_elements": ["Input forms", "Validation system", "Contribution history"]
        },
        {
            "tab": "News & Social Media Sources",
            "filename": "research_figure_7_data_sources.png",
            "purpose": "Data collection overview",
            "key_elements": ["Source statistics", "Collection metrics", "Quality indicators"]
        },
        {
            "tab": "Research Data & Export",
            "filename": "research_figure_8_data_export.png",
            "purpose": "Academic data capabilities",
            "key_elements": ["Export options", "Data formats", "Research citations"]
        }
    ]

    for i, shot in enumerate(screenshots, 1):
        print(f"\n{i}. {shot['tab']}")
        print(f"   📁 File: {shot['filename']}")
        print(f"   🎯 Purpose: {shot['purpose']}")
        print(f"   📊 Key Elements: {', '.join(shot['key_elements'])}")

    print("\n🖥️ HOW TO CAPTURE SCREENSHOTS")
    print("-" * 40)

    print("\n**Windows:**")
    print("1. Press Win + Shift + S")
    print("2. Select area or full screen")
    print("3. Save to 'screenshots' folder")

    print("\n**macOS:**")
    print("1. Press Cmd + Shift + 4")
    print("2. Select area or press Space for window")
    print("3. Save to Desktop or screenshots folder")

    print("\n**Linux:**")
    print("1. Use 'Screenshot' tool or Flameshot")
    print("2. Select area and save")
    print("3. Organize in screenshots directory")

    print("\n💡 CAPTURE BEST PRACTICES")
    print("-" * 40)
    print("• Capture full dashboard interface")
    print("• Include realistic data samples")
    print("• Use high resolution (1920x1080+)")
    print("• Save as PNG format (transparent backgrounds)")
    print("• Name files consistently")
    print("• Add timestamps if showing live data")

    print("\n📁 ORGANIZATION STRUCTURE")
    print("-" * 40)
    print("Create a 'screenshots' folder:")
    print("""
screenshots/
├── research_figures/
│   ├── figure_1_executive_summary.png
│   ├── figure_2_model_performance.png
│   ├── figure_3_sentiment_trends.png
│   └── ...
├── dashboard_tabs/
│   ├── executive_summary_full.png
│   ├── predictions_interface.png
│   └── ...
└── presentations/
    ├── system_overview.png
    └── methodology_demo.png
    """)

    print("\n🎓 RESEARCH INTEGRATION")
    print("-" * 40)
    print("• Add to dissertation Chapter 4")
    print("• Include in appendices as Figures")
    print("• Use in conference presentations")
    print("• Submit with academic publications")
    print("• Demonstrate to thesis committee")

    print("\n📝 FIGURE CAPTIONS EXAMPLES")
    print("-" * 40)
    print("Figure 4.1: GSE Sentiment Analysis Dashboard - Executive Summary")
    print("Figure 4.2: Real-time Price Movement Predictions Interface")
    print("Figure 4.3: Sentiment-Price Correlation Analysis with Granger Causality")
    print("Figure 4.4: Comparative Performance of 12 Machine Learning Algorithms")

    print("\n🚀 QUICK CAPTURE SCRIPT")
    print("-" * 40)
    print("Run this to open dashboard and prepare for screenshots:")
    print("python screenshot_guide.py --open-dashboard")

def open_dashboard():
    """Open dashboard in browser for screenshot capture"""
    dashboard_url = "http://localhost:8501"

    print("🌐 Opening GSE Dashboard for screenshot capture...")
    print(f"📍 URL: {dashboard_url}")

    try:
        webbrowser.open(dashboard_url)
        print("✅ Dashboard opened in browser")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print(f"Please manually open: {dashboard_url}")

    print("\n📸 Ready for screenshot capture!")
    print("Navigate through all tabs and capture screenshots")
    print("Use the guide above for recommended shots")

def create_screenshots_folder():
    """Create organized screenshots folder structure"""
    base_dir = Path("screenshots")
    subdirs = ["research_figures", "dashboard_tabs", "presentations"]

    try:
        for subdir in subdirs:
            (base_dir / subdir).mkdir(parents=True, exist_ok=True)

        print("✅ Created screenshots folder structure:")
        print(f"   📁 {base_dir}/")
        for subdir in subdirs:
            print(f"      ├── {subdir}/")

    except Exception as e:
        print(f"❌ Error creating folders: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--open-dashboard":
        open_dashboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "--create-folders":
        create_screenshots_folder()
    else:
        print_screenshot_guide()
        print("\n" + "=" * 70)
        print("💡 QUICK COMMANDS:")
        print("python screenshot_guide.py --open-dashboard    # Open dashboard")
        print("python screenshot_guide.py --create-folders    # Create folders")
        print("=" * 70)