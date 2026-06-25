# MicroTracker: A Python toolkit for microorganism tracking and trajectory analysis

Interactive tracking software for microscopy videos.

Current features:
- Cell detection
- Frame-to-frame tracking
- Gap closing
- Interactive cell selection
- Live selected-cell tracking
- PyQt GUI

Work in progress.

## Install

```bash
python -m pip install -e .
```

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Notes

Generated analysis results are written to `outputs/`, which is ignored by Git.
Large microscopy videos should stay outside the repository or be added through a
separate data-release workflow.
