@echo off
setlocal
set "PROJECT=%~dp0"
python "%PROJECT%FoldSafe_HybridMDA_Seed2027.py" --data "%PROJECT%data\MDAD" --brmda-view "%PROJECT%data\mdad_brmda_view.npz" --output "%PROJECT%results_seed2027"
endlocal
