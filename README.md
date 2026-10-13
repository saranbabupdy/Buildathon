# VEPL Vegetation Encroachment Prototype

This is a small-scale semantic-segmentation prototype using the VEPL dataset.
It predicts three pixel classes:
- 0: background (RGB mask colour: black)
- 1: vegetation (RGB mask colour: green)
- 2: visible power-line corridor (RGB mask colour: grey)

The model identifies image classes. It does NOT measure electrical clearance or prove a safety violation.

## 1. Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download and extract VEPL
Download the unaugmented tessellated archive from:
https://zenodo.org/records/7800234

Extract it. Locate the directory that contains the image tiles and the directory that contains their corresponding masks. In the VEPL archive these are commonly named with terms such as RGB/Image and Mask.

## 3. Train
Example:
```bash
python train.py --data-dir "C:/data/VEPL/TESELLATED_WITHOUT_AUGMENTATION" --output-dir outputs
```

The script searches for image and mask folders recursively, pairs files by filename stem, and prints the detected folders and number of pairs. If it cannot identify the right folders, use:
```bash
python train.py --image-dir "C:/data/VEPL/RGB" --mask-dir "C:/data/VEPL/Mask" --output-dir outputs
```

Mask encoding expected:
- RGB (0, 0, 0) = background
- RGB (0, 255, 0) = vegetation
- RGB (110, 110, 110) = power-line corridor

These colours are documented in the VEPL paper. The script also maps near-identical colours to the closest expected colour to tolerate minor image encoding differences.

## 4. Predict an image
```bash
python predict.py --model outputs/vepl_unet.keras --image "C:/data/test_drone_image.jpg" --output-dir predictions
```

## 5. Launch dashboard
```bash
streamlit run app.py
```
Upload a drone image and select the trained model file.

## Notes and limitations
- This is a research/demo workflow, not an operational safety system.
- A random image split can overestimate generalization when adjacent tiles come from the same orthomosaic. For formal evaluation, split by original survey/orthomosaic or geographic area.
- The VEPL data has a small number of source orthomosaics. Test on separate drone imagery before claiming real-world generalization.
- Pixel overlap with a corridor class is only a screening signal. The model does not infer unseen wires under vegetation and does not calculate 3D conductor-to-tree distance.
- For real clearance assessment, use suitable calibrated imagery/3D measurements, approved utility clearance rules, and human verification.
- The VEPL dataset is released under CC BY 4.0; follow attribution terms if you share results or derivative work.
