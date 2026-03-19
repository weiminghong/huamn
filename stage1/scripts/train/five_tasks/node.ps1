Set-Location ../../..

$configFile = "configs/qwen_vl_stage1_five_tasks.py"
$outputDir = "Outputs/stage1/qwen_vl_3b_five_tasks/node/$((Get-Date).ToString('MM-dd-yyyy_HH-mm-ss'))"

python tools/train_five_tasks.py `
  --config $configFile `
  --output-dir $outputDir
