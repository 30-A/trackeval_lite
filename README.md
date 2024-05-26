# TrackEval Lite

This is a simpler and more user-friendly implementation of the [TrackEval](https://github.com/JonathonLuiten/TrackEval) script. It evaluates object tracking performance using various metrics.

## Key Differences from the Original TrackEval

TrackEval Lite simplifies the usage and setup process compared to the original TrackEval repository. Here are the key differences:

- **Simplified Input**: Only requires ground truth and tracker result files.
- **No Directory Structure Requirements**: Does not require a specific directory structure or naming style.
- **Direct Sequence Length Input**: Instead of requiring a `seqinfo.ini` file, you directly input the sequence length.
- **Ease of Use**: Streamlined argument handling and setup, focusing on simplicity and ease of use.

## Currently implemented metrics

Metric Family | Sub metrics | Paper | Code |
|-------------|-------------|-------|------|
|             |             |       |      |
|**HOTA metrics**|HOTA, DetA, AssA, LocA, DetPr, DetRe, AssPr, AssRe|[paper](https://link.springer.com/article/10.1007/s11263-020-01375-2)|[code](trackeval/metrics/hota.py)|
|**CLEARMOT metrics**|MOTA, MOTP, MT, ML, Frag, etc.|[paper](https://link.springer.com/article/10.1155/2008/246309)|[code](trackeval/metrics/clear.py)|
|**Identity metrics**|IDF1, IDP, IDR|[paper](https://arxiv.org/abs/1609.01775)|[code](trackeval/metrics/identity.py)|
|**VACE metrics**|ATA, SFDA|[paper](https://link.springer.com/chapter/10.1007/11612704_16)|[code](trackeval/metrics/vace.py)|

## Input Format

The script requires two files: a ground truth file and a tracker result file. Both files should follow this format:

```
<frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>
```

### Example

```
1,1,992,212,57,50
2,1,1002,220,60,55
...
```

## Installation

```bash
git clone https://github.com/30-A/trackeval_lite
cd trackeval_lite
pip install -r requirements.txt
```

## How to Run

```bash
python scripts/run_mot_challenge.py \
  --GT_PATH path/to/gt.txt \
  --TRACKER_PATH path/to/tracker.txt \
  --SEQ_LENGTH 1000 \
  --METRICS HOTA CLEAR VACE Identity
```

## Arguments

The script accepts the following command-line arguments:

- `--GT_PATH` (str): Path to the ground truth file.
- `--TRACKER_PATH` (str): Path to the tracker result file.
- `--SEQ_LENGTH` (int): Length of the sequence in frames.
- `--METRICS` (list): List of metrics to evaluate. Default: `['HOTA', 'CLEAR', 'Identity', 'VACE']`
- `--THRESHOLD` (float): Threshold value for evaluation. Default: `0.5`

## Output

The script generates:

1. **Evaluation Results**: Metrics values printed in a tabular format.
2. **HOTA Curve Plot**: A plot of the HOTA metric over the sequence, saved as `HOTA_curve.png` in the current directory.

## Timing analysis
		
|TrackEval|MOTChallenge|Speedup vs MOTChallenge|py-motmetrics|Speedup vs py-motmetrics
|:---|:---|:---|:---|:---
|9.64|66.23|6.87x|99.65|10.34x

## Citing TrackEval

If you use this code in your research, please use the following BibTeX entry:

```bibtex
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```
