PARTIAL dataset built from the 65 recovered Weed patch boxes (tools/recover_patch_boxes.py -> tools/convert_labelme.py).
- Only ONE weed region per image is labelled; other weeds in the same image are unlabelled (treated as background).
- Crop patches were labelled "Crop" (species unknown) and are therefore NOT included.
- Tomato, Rose, Marigold, Cauliflower have zero instances.
Use it only to test the training / validation pipeline. Do NOT report metrics from it.
