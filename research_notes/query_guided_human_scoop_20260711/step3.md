# Step 3 - Triage
Timestamp: 2026-07-11T07:33:15
VisionLLM v2: LLM super-link to object/keypoint/task decoders; high mechanism overlap.
UniPHD: instance query plus 17 keypoint queries and mask/box/confidence heads; high pose overlap, single referred person, no generative LLM record.
HumanQueryNet: one human query binds all human tasks; exact domain overlap, no LLM.
OMG-LLaVA: object queries connect universal segmentation and LLM; no human pose/attributes.
PixelLM: LLM codebook hidden states feed a pixel decoder for multiple targets; segmentation only.
