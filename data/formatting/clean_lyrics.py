import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tqdm

from pathlib import Path
from argparse import ArgumentParser

from lyric_formatter import *   # Only imports LyricGeniusFormatter class


def main():
    # Args
    arp = ArgumentParser()
    arp.add_argument('-r', '--raw-lyrics-dir',
                     required=True,
                     help='Directory containing raw lyrics_*.json')
    arp.add_argument('-c', '--clean-lyrics-dir',
                     default='./cleanLyrics/',
                     help='Directory to write clean_lyrics_*.json')
    arp.add_argument('-t', '--tokenizer-type',
                     default='bert-base-uncased',
                     choices=['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased'],
                     help='Make sure tokenizer type matches downstream model type!'
                          '(Default = bert-base-uncased)')
    # TODO: Specify train/test/dev split in arp
    opts = arp.parse_args()

    # Glob raw lyrics_*.json
    raw_lyrics_dir = Path(opts.raw_lyrics_dir)
    raw_lyrics_files = sorted(list(raw_lyrics_dir.glob('*.json')))
    print(f'Found {len(raw_lyrics_files)} raw lyrics files')

    # Prepare output paths
    clean_lyrics_dir = Path(opts.clean_lyrics_dir)
    clean_lyrics_dir.mkdir(exist_ok=True)
    train_file = clean_lyrics_dir/f'train-data-{opts.tokenizer_type}.txt'
    dev_file = clean_lyrics_dir/f'dev-data-{opts.tokenizer_type}.txt'
    test_file = clean_lyrics_dir/f'test-data-{opts.tokenizer_type}.txt'
    progress_file = clean_lyrics_dir/f'progress-{opts.tokenizer_type}.txt'

    # Track progress
    progress = list()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as pf:
            for line in pf:
                progress.append(str(line).replace('\n', ''))
    else:
        with open(progress_file, 'a') as _:
            pass

    # Init
    do_lower_case = True if 'uncased' in opts.tokenizer_type else False
    LGF = LyricGeniusFormatter(opts.tokenizer_type, do_lower_case=do_lower_case)

    # Clean text and write to dev, test, and train files
    i = 0
    for rlf in tqdm.tqdm(raw_lyrics_files):
        if str(rlf) not in progress:
            formatted_song = LGF.format_lyrics(rlf)

            if formatted_song:
                i += 1

                # 5% to dev
                if i % 20 == 0:
                    with open(dev_file, 'a') as df:
                        for section in formatted_song['sections']:
                            for l in section:
                                df.write(f'{l} [SEP] \n')
                            df.write('\n')  # Double newline between sections

                # 5% to test
                elif i % 20 == 1:
                    with open(test_file, 'a') as tef:
                        for section in formatted_song['sections']:
                            for l in section:
                                tef.write(f'{l} [SEP] \n')
                            tef.write('\n')

                # 90% to train
                else:
                    with open(train_file, 'a') as trf:
                        for section in formatted_song['sections']:
                            for l in section:
                                trf.write(f'{l} [SEP] \n')
                            trf.write('\n')

            # Record progress
            progress.append(str(rlf))
            with open(progress_file, 'a') as pf:
                pf.write(f'{rlf}\n')


if __name__ == '__main__':
    main()
