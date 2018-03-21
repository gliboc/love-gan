# vega

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
