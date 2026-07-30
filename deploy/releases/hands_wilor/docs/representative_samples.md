# Representative Dataset Samples

This sample set is for pipeline design and visual QC. It deliberately avoids using the tomato kitchen clip as the only evidence source.

| Class | Clip | Frames | Why It Matters |
| --- | --- | ---: | --- |
| Long tool, floor cleaning | `/data2/egoscale_demo_30h/egoscale_tasks/20251210_0002_Rec4afc_P0_S296a7f_task_4/20251210_0002_Rec4afc_P0_S296a7f_task_4.mp4` | 4481 | Long mop geometry, large camera motion, tool spans much of the image. A hand-only or small-object pipeline cannot explain the manipulated object. |
| Long tool, bedroom | `/data2/egoscale_demo_30h/egoscale_tasks/20260112_1458_Recc175_P0_S46092f_task_5/20260112_1458_Recc175_P0_S46092f_task_5.mp4` | 2190 | Same tool class in a different room and lighting regime, useful for checking whether object logic overfits to kitchen color cues. |
| Fine dexterity | `/data2/egoscale_demo_30h/egoscale_tasks/20251212_0831_Recb7ab_P1_Sd9bd5d_task_1/20251212_0831_Recb7ab_P1_Sd9bd5d_task_1.mp4` | 5400 | Crochet hook and yarn are small, deformable, and heavily occluded by fingers. Contact priors matter more than category detectors. |
| Deformable clothing | `/data2/egoscale_demo_30h/egoscale_tasks/20260213_1412_Recfb8a_P0_Sf0c9d6_task_10/20260213_1412_Recfb8a_P0_Sf0c9d6_task_10.mp4` | 1500 | Clothing changes shape and has weak rigid-pose semantics. A rigid 6D pose stage must switch to surface or keypoint state here. |
| Rigid tabletop objects | `/data2/egoscale_demo_30h/egoscale_tasks/20260201_2103_Reca4be_P0_S0cc0b2_task_5/20260201_2103_Reca4be_P0_S0cc0b2_task_5.mp4` | 1890 | Keyboard/tabletop manipulation tests multi-object selection and occlusion on a non-kitchen workspace. |
| Clutter and thin objects | `/data2/egoscale_demo_30h/egoscale_tasks/20251224_1141_Rec3a3b_P0_S3a8b63_task_4/20251224_1141_Rec3a3b_P0_S3a8b63_task_4.mp4` | 1350 | Plants, branches, and tools create thin structures near hands; segmentation masks can merge with clutter. |
| Deformable bag | `/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4` | 1050 | Trash bag and bin interactions stress large deformable object tracking and hand/object occlusion. |

Contact sheets are stored under `/data2/ego_annotation_outputs/representative_preview/`.

The object annotation stage must choose the manipulated object from hand contact, temporal persistence, and semantic action context. A tomato-only color rule is useful only as a debug signal on the kitchen clip and cannot be a pipeline component for this dataset.

## Current Representative Run

Trash-bag run:

`/data2/ego_annotation_outputs/representative_trash/fused_bagprompt_full_final/`

- DROID: 1050/1050 dense camera poses.
- WiLoR: hands detected in 898/1050 frames.
- Object front-end: action-segment `trash_bag` profile, separated from `trash_can`; OWLv2 prompt proposals; SAM masks; hand-contact, temporal, and deformable-size rejection.
- Videos: 1050 frames at 30 fps; overlay/reconstruction 960x540; side-by-side 1920x540.
- Object track: 807 measured frames, 3 predicted frames, 13 rejected invalid/degenerated measurements.
- World object fusion: 810 active world states, 585 DROID-depth frames, 321 contact-anchor frames.
- Hand/object contact correction: 819 accepted contact-depth measurements and 775 corrected hand frames.
- Renderer: head-local world-coordinate view with an explicit `HEAD CAM` frustum and deformable object surface patch rather than a sphere-only proxy.

Inspected frames: 90, 269, 678, 900, 910, and 917. Frames 90, 269, and 678 have measured bag states. Frame 900 keeps a large predicted bag surface while the target region is still visible. Frames 910 and 917 are marked unobserved and draw no object, avoiding the earlier false plant-side object state.

The deformable-bag result is now a surface proxy, not a rigid 6D pose. That representation is appropriate for this clip because the manipulated state is a bag surface being opened and lined around a bin. The remaining limitation is metric depth and surface shape accuracy: the object surface comes from mask rays plus optimized depth/contact anchors, without depth sensors, CAD, fiducials, or ground-truth scale.
