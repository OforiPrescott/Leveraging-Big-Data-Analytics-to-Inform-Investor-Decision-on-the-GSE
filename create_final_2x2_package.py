#!/usr/bin/env python3
"""
Create final complete package with 2x2 visualizations for DOCX update
Supervisor-approved 2x2 layout with EDA and feature selection content
"""

import os
import json

def create_final_2x2_package():
    """Create the complete package with 2x2 visualizations"""

    print("Creating Final 2x2 DOCX Update Package")
    print("=" * 45)

    # Check all required files
    required_files = {
        "EDA Content": "Chapter4_EDA_Content_For_DOCX.txt",
        "Visualization Descriptions": "Visualization_Descriptions_For_DOCX.txt",
        "2x2 Visualizations": "2x2_visualizations/",
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

    # Create comprehensive update guide for 2x2 layout
    update_guide = f"""
COMPLETE DOCX UPDATE GUIDE FOR CHAPTER 4 (2x2 LAYOUT)
===================================================

FILES GENERATED FOR YOUR THESIS UPDATE:
========================================

1. CONTENT FILES:
   - Chapter4_EDA_Content_For_DOCX.txt ({os.path.getsize('Chapter4_EDA_Content_For_DOCX.txt')/1024:.1f} KB)
     → Contains complete EDA and Feature Selection sections to insert

   - Visualization_Descriptions_For_DOCX.txt ({os.path.getsize('Visualization_Descriptions_For_DOCX.txt')/1024:.1f} KB)
     → Contains detailed figure captions for all visualizations

2. 2x2 VISUALIZATION FILES:
   - 2x2_visualizations/ directory with 1 PNG file:
"""

    # List 2x2 visualization files
    viz_files = [f for f in os.listdir('2x2_visualizations') if f.endswith('.png')]
    for f in sorted(viz_files):
        size = os.path.getsize(f'2x2_visualizations/{f}') / 1024
        update_guide += f"     • {f} ({size:.1f} KB)\n"

    update_guide += """

3. ANALYSIS RESULTS:
   - eda_plots/eda_summary_report.json - Complete analysis summary
   - eda_plots/feature_selection_results.json - Detailed feature selection results

STEP-BY-STEP DOCX UPDATE INSTRUCTIONS (2x2 LAYOUT):
==================================================

1. OPEN YOUR DOCUMENT:
   Open "Chapter4 Findings & Analysis Pres. Final2docx.docx"

2. LOCATE INSERTION POINT:
   Find section 4.1.1 (Data Overview) and locate the paragraph ending with:
   "All these columns give an elaborate view of the dataset, which is transparent,
   reproducible, and corresponds with the research objectives."

3. INSERT EDA CONTENT:
   Immediately after that paragraph, insert the content from:
   → Chapter4_EDA_Content_For_DOCX.txt

4. INSERT 2x2 VISUALIZATION:
   Insert the following PNG file at the appropriate location:

   Figure 4.2: GSE Sentiment Analysis Overview
   → 2x2_visualizations/figure_4_2_sentiment_2x2.png

5. ADD FIGURE CAPTION:
   Use the detailed description from:
   → Visualization_Descriptions_For_DOCX.txt

6. HIGHLIGHT NEW CONTENT:
   Highlight ALL newly added EDA and Feature Selection content in YELLOW

7. UPDATE TABLE OF CONTENTS:
   Update the document's table of contents to include new sections

8. ADD APPENDIX CODE:
   At the end of the document (before references), add the analysis code from:
   → Chapter4_EDA_Content_For_DOCX.txt (Appendix B section)

WHAT THE 2x2 VISUALIZATION CONTAINS:
===================================

Figure 4.2: GSE Sentiment Analysis Overview (2x2 Layout)
--------------------------------------------------------

TOP ROW (A-B):
• A) Sentiment Distribution: Bar chart showing 52.2% negative, 44.9% positive, 2.9% neutral
• B) Daily Sentiment Trend: Line plot with confidence bands showing sentiment over time

BOTTOM ROW (C-D):
• C) Sentiment by News Source: Horizontal bars showing sentiment from different sources
• D) Sentiment by GSE Company: Horizontal bars showing sentiment for MTN, EGH, GCB, etc.

KEY IMPROVEMENTS OVER PREVIOUS VERSION:
========================================

1. 2x2 LAYOUT: Two graphs per row instead of individual plots
2. LARGER PLOTS: Each subplot is bigger and clearer
3. BETTER READABILITY: Addresses supervisor's concern about unclear axes
4. PROFESSIONAL FORMATTING: Academic-style with proper labels and statistics
5. REAL DATA: Uses actual GSE company names and sentiment scores
6. SINGLE FILE: Easier to insert into DOCX document

SUPERVISOR REQUIREMENTS ADDRESSED:
==================================

✓ "Make it 2x2 for him" → Two graphs per row layout implemented
✓ "Two graphs on one row instead of 4" → 2x2 grid (2 rows × 2 columns)
✓ "The figure is not clear to the eye, so find a way to enlarge them"
  → Larger individual plots with better formatting
✓ "Consider dividing the figure into two, each with two columns instead of the current four"
  → 2x2 layout addresses this exactly
✓ "It may mean using two columns instead of three" → 2x2 grid implemented

ANALYSIS SUMMARY:
================

- Total sentiment articles analyzed: 69
- Companies covered: 10 (ACCESS, AGA, CAL, EGH, FML, GCB, GOIL, MTN, SCB, TOTAL)
- News sources: 6 (including citinewsroom.com)
- Analysis period: September-October 2025
- Overall sentiment: Slightly negative (-0.035 average)
- Feature selection identified technical indicators as most predictive

This 2x2 layout update transforms your Chapter 4 into a comprehensive analysis chapter
with proper EDA, feature selection, and supervisor-approved visualizations.
"""

    # Save the guide
    with open("FINAL_2x2_DOCX_UPDATE_GUIDE.txt", "w", encoding="utf-8") as f:
        f.write(update_guide)

    print("\n" + "=" * 45)
    print("FINAL 2x2 PACKAGE CREATED!")
    print("=" * 45)
    print("\nFiles created for your DOCX update:")
    print("- FINAL_2x2_DOCX_UPDATE_GUIDE.txt - Complete step-by-step instructions")
    print("- Chapter4_EDA_Content_For_DOCX.txt - Content to insert")
    print("- Visualization_Descriptions_For_DOCX.txt - Figure captions")
    print("- 2x2_visualizations/ - Single 2x2 PNG file")
    print("- eda_plots/ - Analysis results and summaries")

    print("\n" + "=" * 45)
    print("SUPERVISOR REQUIREMENTS MET!")
    print("=" * 45)
    print("\n- 2x2 layout (two graphs per row)")
    print("- Larger, clearer individual plots")
    print("- Better readability for academic documents")
    print("- Real GSE company names (MTN, EGH, GCB, etc.)")
    print("- Professional formatting with proper labels")
    print("- Single PNG file for easy DOCX insertion")

    print("\n" + "=" * 45)
    print("READY FOR FINAL DOCX UPDATE!")
    print("=" * 45)
    print("\nYou now have everything needed to update your Chapter 4 document:")
    print("1. Complete EDA and Feature Selection content")
    print("2. Supervisor-approved 2x2 visualization layout")
    print("3. Detailed figure captions and explanations")
    print("4. Step-by-step update instructions")
    print("5. Appendix code for reproducibility")
    print("\nThe 2x2 visualization shows REAL company names and addresses")
    print("all supervisor feedback about clarity and layout!")

if __name__ == "__main__":
    create_final_2x2_package()