# Step 5 - Full-Paper Checks
Timestamp: 2026-07-11T07:33:15
VisionLLM v2: super link transfers LLM task information and gradients to task-specific decoders; supports object localization and pose. Shared object/keypoint queries can hurt keypoint AP, motivating task-specific child queries.
UniPHD: each query group contains one prompt-conditioned instance query and 17 keypoint queries; deformable attention captures local detail and graph attention models joint-instance dependencies. Losses: box L1 5, GIoU 2, class focal 2, pose L1 10, OKS 4, visibility CE 4, mask Dice 5, mask focal 2.
HumanQueryNet: one matched Human Query feeds box, keypoint, mask, and attribute heads.
