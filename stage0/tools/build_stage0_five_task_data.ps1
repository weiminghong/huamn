$inputJsonl = "data/human/unified/all.jsonl"
$outputDir = "data/human/stage1_five_tasks"

python stage0/tools/build_stage0_five_task_data.py `
  --input-jsonl $inputJsonl `
  --output-dir $outputDir
