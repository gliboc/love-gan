# vega

## Requirements

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data set
- For DCGAN, we used the [11k Hands dataset](https://sites.google.com/view/11khands).
  To set up the dataset to the expected picture size `32x32`,
  unzip the dataset in the folder `input/hands_orig`.
  Then generate the smaller images with our custom script with the following command:

  ```
  mkdir input/hands_32
  python3 resize.py 32 input/hands_orig input/hands_32
  ```

- For InfoGAN, we used the [Fruits 360 dataset](https://www.kaggle.com/moltean/fruits).
  To set up the dataset to the expected pictures size `96x96`,
  unzip the dataset in the folder `input`.
  Then, merge the content of the different folders in a single folder: `input/fruits_orig`.
  Actually, you can choose how many directories you use, and set the associated number of 
  classes you expect inside the `CONFIG` table, in `infogan.py`.
  You can then generate the images using:
  
  ```
  mkdir input/fruits_96
  python3 resize.py 96 input/fruits_orig input/fruits_96
  ```

## Architecture 

- cache: Preprocessed datasets that don’t need to be re-generated every time you perform an analysis.
- config: Configuration settings for the project
- data: Raw data files.
- munge: Preprocessing data munging code, the outputs of which are put in cache.
- src: Statistical analysis scripts.
- doc: Documentation written about the analysis.
- graphs: Graphs created from analysis.
- logs: Output of scripts and any automatic logging.
- profiling: Scripts to benchmark the timing of your code.
- reports: Output reports and content that might go into reports such as tables.
- tests: Unit tests and regression suite for your code.
