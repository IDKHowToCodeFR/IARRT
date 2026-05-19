@echo off
echo ========================================================
echo IARRT Research Suite: SOTA Pipeline Execution
echo ========================================================

echo [1/4] Expanding Idiom Knowledge Base...
python data\load_data.py

echo [2/4] Training Robust Idiom Detector (BERT)...
python train_detector.py

echo [3/4] Running Comprehensive Baseline Comparison (Paper-Grade)...
python compare_baselines.py --eval-size 30

echo [4/4] Generating SOTA Visualizations...
python visualize_research.py

echo ========================================================
echo Research Pipeline Complete!
echo Results available in the 'outputs/' directory:
echo   - baseline_comparison.csv (Full Metrics)
echo   - research_impact_radar.png (Visual Superiority)
echo   - error_reduction_analysis.png (Metric Deep Dive)
echo   - model_confidence_distribution.png (Routing Analysis)
echo ========================================================
pause
