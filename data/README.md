# Dataset Datasheet
This README documents the process of rap lyric acquisition and preparing the dataset for fine-tuning a BERT model. In the final section, the data itself is explored and visualized. 

## Data Acquisition:
* Artist List: [Wikipedia Top Hip Hop Musicians](https://en.wikipedia.org/wiki/List_of_hip_hop_musicians)
* Lyrics Source: [Genius.com](https://genius.com)
* Means of Acquisition: [LyricsGenius pip package](https://github.com/johnwmillr/LyricsGenius)

Notes to be aware of:  
* Sometimes the artist names from Wikipedia's list are automatically changed by LyricsGenius
    - 80% of the time, it trivially changes punctuation to find the correct artist index, as one would expect 
    - 20% of the time, it finds a similar(ish) name of a verifiably different artist (i.e. F(something)??? --> Fall Out Boy (manually deleted with `rm lyrics_falloutboy_*.json`))
* Using `excluded_terms = ['(Remix)', '(Live)', '(Translation)']` automatically skips songs with *live* anywhere in the song title string 
    - 70% of the time, it skips remixes and interviews as expected
    - 30% of the time, a song is skipped erroneously because a word like 'Alive' is in it

## Data Cleaning & Tokenization:

## Artist Vocabulary Analysis:
Reference `data/WikipediaRapArtists.txt` for full, alphabetized list of artists in lyrics dataset. 

### Gucci Mane
* n Songs: 
* Release Dates:
* Vocabulary Size: 
* Term Frequency Histogram: 
* Key Term Dispersion Plot: 

### Eminem
* n Songs: 
* Release Dates:
* Vocabulary Size: 
* Term Frequency Histogram: 
* Key Term Dispersion Plot: 

### Jay Z
* n Songs: 
* Release Dates:
* Vocabulary Size: 
* Term Frequency Histogram: 
* Key Term Dispersion Plot: 
