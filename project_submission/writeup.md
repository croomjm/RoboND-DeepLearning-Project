|  | Learning Rate | Batch Size | Epochs | Steps per Epoch | Validation Steps | Optimizer | IOU | Score | Dataset |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.005 | 256 | 12 | 500 | 50 | Nadam | 0.562 | 0.411 | Suplemental |
| 2 | 0.005 | 64 | 15 | 400 | 50 | Nadam | 0.535 | 0.405 | Baseline |
| 3 | 0.005 | 64 | 12 | 400 | 50 | Nadam | 0.541 | 0.398 | Supplemental |
| 4 | 0.005 | 128 | 15 | 400 | 50 | Adam | 0.537 | 0.392 | Baseline |
| 5 | 0.01 | 256 | 15 | 500 | 50 | Nadam | 0.529 | 0.390 | Baseline |
| 6 | 0.005 | 64 | 15 | 400 | 50 | Adam | 0.539 | 0.389 | Baseline |
| 7 | 0.002 | 64 | 15 | 500 | 50 | Adam | 0.535 | 0.386 | Baseline |
| 8 | 0.01 | 256 | 15 | 500 | 50 | Adam | 0.518 | 0.377 | Baseline |
| 9 | 0.005 | 128 | 12 | 400 | 50 | Nadam | 0.520 | 0.374 | Supplemental |
| 10 | 0.01 | 128 | 15 | 500 | 50 | Adam | 0.506 | 0.370 | Baseline |
| 11 | 0.01 | 64 | 15 | 500 | 50 | Adam | 0.505 | 0.363 | Baseline |
| 12 | 0.002 | 128 | 15 | 500 | 50 | Adam | 0.501 | 0.361 | Baseline |


|  | Overall<br>Score | Following Target<br>Percent False Negatives | No Target<br>Percent False Positives | Far from Target<br>Percent False Positives | Far from Target<br>Percent False Negatives | Dataset |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.411 | 0.0% | 27.306% | 0.619% | 52.632% | Suplemental |
| 2 | 0.405 | 0.0% | 16.236% | 0.31% | 52.941% | Baseline |
| 3 | 0.398 | 0.0% | 26.199% | 0.929% | 51.703% | Suplemental |
| 4 | 0.392 | 0.0% | 18.819% | 0.31% | 58.204% | Baseline |
| 5 | 0.39 | 0.0% | 12.915% | 0.619% | 60.062% | Baseline |
| 6 | 0.389 | 0.0% | 23.985% | 0.929% | 57.276% | Baseline |
| 7 | 0.386 | 0.0% | 28.413% | 0.31% | 55.108% | Baseline |
| 8 | 0.377 | 0.0% | 16.974% | 0.619% | 60.372% | Baseline |
| 9 | 0.374 | 0.0% | 19.188% | 0.31% | 60.991% | Suplemental |
| 10 | 0.37 | 0.0% | 10.332% | 0.31% | 63.158% | Baseline |
| 11 | 0.363 | 0.184% | 16.236% | 0.31% | 62.539% | Baseline |
| 12 | 0.361 | 0.0% | 13.653% | 0.31% | 64.396% | Baseline |
