# Local datasets

Data and camera images are excluded from Git. Obtain datasets through their authorized channels.

- `dataset.csv`: synthetic smoke-test samples; not evidence of real-world accuracy.
- `ck_plus_landmarks.csv`: local extracted CK+ features, 852 samples / 106 subjects; no neutral class.
- `ck_raw.parquet`: local source images. Not redistributed.

For a usable neutral class, collect real samples from multiple consenting subjects:
`python collect_and_train.py --action collect --subject person01 --csv data/custom.csv`
Use a new subject ID for each person and repeat across lighting, head poses and sessions.
