from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve

from random import random, randrange, randint, shuffle, choice, sample
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import json


class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = list()
    last_indices = list()
    for i, token in enumerate(tokens):
        if token == '[CLS]':
            continue
        elif token == '[SEP]':
            last_indices.append(i-1)
        else:
            cand_indices.append(i)

    # Delete copies from cand_indices
    for i in last_indices:
        if i in cand_indices:
            cand_indices.pop(cand_indices.index(i))

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))

    # Sample both 40/60
    shuffle(last_indices), shuffle(cand_indices)
    # Do not sample more than len(indices)
    mask_indices = sample(last_indices, min(int(num_to_mask * 0.4), len(last_indices)))
    mask_indices.extend(sample(cand_indices, min(int(num_to_mask * 0.6), len(cand_indices))))
    mask_indices.sort()

    masked_token_labels = list()
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = '[MASK]'
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS] ([SEP] tokens were already added between each line)
    max_num_tokens = max_seq_length - 3
    # Sometimes [SEP] can be double truncated so 3 is necessary
    # NOTE: tokens_a and tokens_b come from different sections

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(10, max_num_tokens)     # Higher min seq len for higher short seq prob

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments 'A' and 'B' based on the actual 'sentences' provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                # Random next
                if len(current_chunk) == 1 or random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we 'put them back' so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # NOTE: [SEP] tokens are added after each line during data cleaning step,
                #       but they may have been truncated
                tokens_a = tokens_a + ['[SEP]'] if '[SEP]' not in tokens_a[-1] else tokens_a
                tokens_b = tokens_b + ['[SEP]'] if '[SEP]' not in tokens_b[-1] else tokens_b
                tokens = ['[CLS]'] + tokens_a + tokens_b
                # Range +1 [CLS] token
                segment_ids = [0 for _ in range(len(tokens_a) + 1)] + [1 for _ in range(len(tokens_b))]
                assert len(tokens) == len(segment_ids), 'Len mismatch'

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)

                instance = {
                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'is_random_next': is_random_next,
                    'masked_lm_positions': masked_lm_positions,
                    'masked_lm_labels': masked_lm_labels}
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--bert_model', type=str, required=True,
                        choices=['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased',
                                 'bert-base-multilingual', 'bert-base-chinese'])
    parser.add_argument('--do_lower_case', action='store_true')

    parser.add_argument('--reduce_memory', action='store_true',
                        help='Reduce memory usage for large datasets by keeping data on disc rather than in memory')

    parser.add_argument('--epochs_to_generate', type=int, default=3,
                        help='Number of epochs of data to pregenerate')
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--short_seq_prob', type=float, default=0.1,
                        help='Probability of making a short sentence as a training example')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15,
                        help='Probability of masking each token for the LM task')
    parser.add_argument('--max_predictions_per_seq', type=int, default=20,
                        help='Maximum number of tokens to mask in each sequence')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            for line in tqdm(f, desc='Loading Dataset', unit=' lines'):
                line = line.strip()
                if line == '':
                    docs.add_document(doc)
                    doc = []
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            exit('ERROR: No document breaks were found in the input file! These are necessary to allow the script to '
                 'ensure that random NextSentences are not sampled from the same document. Please add blank lines to '
                 'indicate breaks between documents in your input file. If your dataset does not contain multiple '
                 'documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, '
                 'sections or paragraphs.')

        args.output_dir.mkdir(exist_ok=True)
        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            epoch_filename = args.output_dir / f'epoch_{epoch}.json'
            num_instances = 0
            with epoch_filename.open('w') as epoch_file:
                for doc_idx in trange(len(docs), desc='Document'):
                    doc_instances = create_instances_from_document(
                        docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        vocab_list=vocab_list)
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = args.output_dir / f'epoch_{epoch}_metrics.json'
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    'num_training_examples': num_instances,
                    'max_seq_len': args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()
