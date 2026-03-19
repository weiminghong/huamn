Set-Location ../../..

$configFile = "configs/qwen_vl_stage1_five_tasks.py"
$checkpoint = "Outputs/stage1/qwen_vl_3b_five_tasks"
$split = "val"

python tools/test_five_tasks.py `
  --config $configFile `
  --checkpoint $checkpoint `
  --split $split
