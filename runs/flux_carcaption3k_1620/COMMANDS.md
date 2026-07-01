# FLUX CarCaption3K-1620 finetune — launch commands

## Prerequisite: materialize the (gitignored) training dataset first
# The config's folder_path points to a symlink dataset that is NOT in git.
# Rebuild it from the locked manifest before launching:
cd /home/damian/Projects/quality_assessment_library
/home/damian/miniconda3/envs/trellis_qa/bin/python scripts/build_flux_filtered_dataset.py --splits train,val
# -> data/ablation/carcaption3k_1620_flux_train/  (1620 <sha>.png + <sha>.txt symlinks)

## FLUX finetune (run on a GPU box, ~24GB+ VRAM)
cd /home/damian/Projects/ai-toolkit
python run.py config/flux_trellis_carcaption3k_1620.yaml
# Output LoRA -> ai-toolkit/output/flux_carcaption3k_1620_lora32/
# Budget: 8000 steps, LoRA rank 32, lr 1e-4 (matched to MeshFleet FLUX; see runs/reference_budgets.json)
