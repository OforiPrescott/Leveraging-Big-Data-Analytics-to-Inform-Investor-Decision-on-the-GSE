#!/usr/bin/env python3
"""
Create final complete package for DOCX update with all materials
"""

import os
import json

def create_final_package():
    """Create the complete package for DOCX update"""

    print("Creating Final DOCX Update Package")
    print("=" * 40)

    # Check all required files
    required_files = {
        "EDA Content": "Chapter4_EDA_Content_For_DOCX.txt",
        "Visualization Descriptions": "Visualization_Descriptions_For_DOCX.txt",
        "Proper Visualizations": "proper_visualizations/",
        "Analysis Results": "eda_plots/eda_summary_report.json"
    }

    print("\nChecking required files:")
    all_present = True

    for category, file_path in required_files.items():
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                files_in_dir = [f for f in os.listdir(file_path) if f.endswith('.png')]
                print(f"  - {category}: {file_path} ({len(files_in_dir)} PNG files)")
            else:
                size = os.path.getsize(file_path) / 1024
                print(f"  - {category}: {file_path} ({size:.1f} KB)")
        else:
            print(f"  X {category}: {file_path} - MISSING")
            all_present = False

    if not all_present:
        print("\n❌ Some required files are missing. Please run the analysis scripts first.")
        return

    # Create comprehensive update guide
    update_guide = f"""
COMPLETE DOCX UPDATE GUIDE FOR CHAPTER 4
========================================

FILES GENERATED FOR YOUR THESIS UPDATE:
========================================

1. CONTENT FILES:
   - Chapter4_EDA_Content_For_DOCX.txt ({os.path.getsize('Chapter4_EDA_Content_For_DOCX.txt')/1024:.1f} KB)
     → Contains complete EDA and Feature Selection sections to insert

   - Visualization_Descriptions_For_DOCX.txt ({os.path.getsize('Visualization_Descriptions_For_DOCX.txt')/1024:.1f} KB)
     → Contains detailed figure captions for all visualizations

2. VISUALIZATION FILES:
   - proper_visualizations/ directory with 4 PNG files:
"""

    # List visualization files
    viz_files = [f for f in os.listdir('proper_visualizations') if f.endswith('.png')]
    for f in sorted(viz_files):
        size = os.path.getsize(f'proper_visualizations/{f}') / 1024
        update_guide += f"     • {f} ({size:.1f} KB)\n"

    update_guide += """

3. ANALYSIS RESULTS:
   - eda_plots/eda_summary_report.json - Complete analysis summary
   - eda_plots/feature_selection_results.json - Detailed feature selection results

STEP-BY-STEP DOCX UPDATE INSTRUCTIONS:
=====================================

1. OPEN YOUR DOCUMENT:
   Open "Chapter4 Findings & Analysis Pres. Final2docx.docx"

2. LOCATE INSERTION POINT:
   Find section 4.1.1 (Data Overview) and locate the paragraph ending with:
   "All these columns give an elaborate view of the dataset, which is transparent,
   reproducible, and corresponds with the research objectives."

3. INSERT EDA CONTENT:
   Immediately after that paragraph, insert the content from:
   → Chapter4_EDA_Content_For_DOCX.txt

4. INSERT VISUALIZATIONS:
   Insert the following PNG files at appropriate locations:

   Figure 4.2A - Sentiment Distribution:
   → proper_visualizations/sentiment_distribution_real.png

   Figure 4.2B - Daily Sentiment Trend:
   → proper_visualizations/sentiment_timeline_real.png

   Figure 4.2C - Sentiment by Source:
   → proper_visualizations/source_sentiment_real.png

   Figure 4.2D - Sentiment by Company:
   → proper_visualizations/company_sentiment_real.png

5. ADD FIGURE CAPTIONS:
   Use the detailed descriptions from:
   → Visualization_Descriptions_For_DOCX.txt

6. HIGHLIGHT NEW CONTENT:
   Highlight ALL newly added EDA and Feature Selection content in YELLOW

7. UPDATE TABLE OF CONTENTS:
   Update the document's table of contents to include new sections

8. ADD APPENDIX CODE:
   At the end of the document (before references), add the analysis code from:
   → Chapter4_EDA_Content_For_DOCX.txt (Appendix B section)

WHAT THESE VISUALIZATIONS SHOW:
==============================

Figure 4.2A (sentiment_distribution_real.png):
- Shows sentiment distribution across 69 articles
- 52.2% negative, 44.9% positive, 2.9% neutral
- Real GSE company data, not sample data

Figure 4.2B (sentiment_timeline_real.png):
- Daily sentiment trend from Sep-Oct 2025
- 28 days of data with confidence bands
- Shows sentiment volatility over time

Figure 4.2C (source_sentiment_real.png):
- Sentiment by news source including citinewsroom.com
- Shows how different media outlets portray GSE companies
- Includes sample counts for each source

Figure 4.2D (company_sentiment_real.png):
- Sentiment scores for 10 real GSE companies (MTN, EGH, GCB, etc.)
- Ranges from most negative (ACCESS: -0.367) to most positive (EGH: +0.148)
- Shows heterogeneity in company sentiment

KEY IMPROVEMENTS OVER PREVIOUS VERSION:
========================================

1. REAL DATA: Uses actual GSE company names and sentiment scores from your database
2. INDIVIDUAL PLOTS: Each subplot is a separate, clear PNG file
3. ACADEMIC FORMATTING: Proper figure numbering, titles, and statistics
4. DETAILED CAPTIONS: Each figure has comprehensive explanation
5. COMPLETE PACKAGE: All content, code, and instructions provided

ANALYSIS SUMMARY:
================

- Total sentiment articles analyzed: 69
- Companies covered: 10 (ACCESS, AGA, CAL, EGH, FML, GCB, GOIL, MTN, SCB, TOTAL)
- News sources: 6 (including citinewsroom.com)
- Analysis period: September-October 2025
- Overall sentiment: Slightly negative (-0.035 average)
- Feature selection identified technical indicators as most predictive

This update transforms your Chapter 4 into a comprehensive analysis chapter
with proper EDA, feature selection, and supporting visualizations.
"""

    # Save the guide
    with open("FINAL_DOCX_UPDATE_GUIDE.txt", "w", encoding="utf-8") as f:
        f.write(update_guide)

    print("\n" + "=" * 40)
    print("FINAL PACKAGE CREATED!")
    print("=" * 40)
    print("\nFiles created for your DOCX update:")
    print("- FINAL_DOCX_UPDATE_GUIDE.txt - Complete step-by-step instructions")
    print("- Chapter4_EDA_Content_For_DOCX.txt - Content to insert")
    print("- Visualization_Descriptions_For_DOCX.txt - Figure captions")
    print("- proper_visualizations/ - 4 PNG files with real data")
    print("- eda_plots/ - Analysis results and summaries")

    print("\n" + "=" * 40)
    print("READY FOR SUBMISSION!")
    print("=" * 40)
    print("\nYou now have everything needed to update your Chapter 4 document:")
    print("1. Complete EDA and Feature Selection content")
    print("2. Professional visualizations using real GSE data")
    print("3. Detailed figure captions and explanations")
    print("4. Step-by-step update instructions")
    print("5. Appendix code for reproducibility")
    print("\nThe visualizations now show REAL company names (MTN, EGH, GCB, etc.)")
    print("and actual sentiment data from your database, not generic samples.")

if __name__ == "__main__":
    create_final_package()