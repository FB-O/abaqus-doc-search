import concurrent.futures
import logging
import math

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Allow debug logs to propagate


class PreProcessor:
    """
    - Chunk sentences in parallel
    - (that's all the preprocessing for now)

    Example:
    -------
        >>> pre = PreProcessor(4, 0.2)
        >>> ids = [1, 2, 3]
        >>> sentences = ['This is a test sentence.', 'Short sentence.', 'This is a rather very long sentence, phew!']
        >>> chunk_ids, chunks = pre(ids, sentences)
        >>> chunk_ids
        [1, 1, 2, 3, 3, 3]
        >>> chunks
        [
            'This is a test',
            'test sentence.',
            'Short sentence.',
            'This is a rather,
            'rather very long sentence,
            'sentence, phew!'
        ]
    """
    def __init__(self, max_chunk_len: int, overlap_p: float, max_workers: int = None):
        """
        overlap: percentage of max_chunk_len.
        max_workers: maximum number of workers for parallel processing. None = automatic based on CPU count.
        """
        self.max_chunk_len = max_chunk_len
        self.overlap = math.ceil(overlap_p * max_chunk_len)
        self.max_workers = max_workers

    def __call__(self, ids: Iterable[int], sentences: Iterable[str]):
        """Return chunks processed in parallel"""
        # Convert to lists if they aren't already
        ids_list = list(ids)
        sentences_list = list(sentences)

        # Process sentences in parallel
        all_chunks_ids = []
        all_chunks = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Map _chunk_sentence across all id-sentence pairs
            results = list(executor.map(self._chunk_sentence, ids_list, sentences_list))

        # Unpack results
        for ids_, chunks_ in results:
            all_chunks_ids.extend(ids_)
            all_chunks.extend(chunks_)

        return all_chunks_ids, all_chunks

    def _chunk_sentence(self, id, sentence):
        if sentence is None:
            return [id], ['']
        sentence = sentence.split(' ')
        if len(sentence) < self.max_chunk_len:
            return [id], [' '.join(sentence)]
        else:
            n_chunks = math.ceil((len(sentence) - self.overlap) / (self.max_chunk_len - self.overlap))
            ids = [id] * n_chunks
            words = []
            for i in range(n_chunks):
                start = i * (self.max_chunk_len - self.overlap)
                end = start + self.max_chunk_len
                words.append(sentence[start: end])
            return ids, [' '.join(word) for word in words]

class PostProcessor:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

    def __call__(self, encoded_input, model_output):
        if self.model_name == 'sentence-transformers/all-MiniLM-L6-v2':
            #Mean Pooling - Take attention mask into account for correct averaging
            attention_mask = encoded_input['attention_mask'].to(self.device)
            token_embeddings = model_output[0].to(self.device) #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-1)
        else:
            print(f"Model {self.model_name} not supported.")


class SentenceEmbedder:
    def __init__(self, model_name: str, device: str, chunk_params: dict = {}):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.preprocessor = self._set_preprocessor(chunk_params)
        self.postprocessor = PostProcessor(model_name, self.device)

    def _set_preprocessor(self, chunk_params: dict):
        if not chunk_params.get('max_chunk_len'):
            chunk_params['max_chunk_len'] = self.model.config.max_position_embeddings // 2
        if not chunk_params.get('overlap_p'):
            chunk_params['overlap_p'] = 0.25
        preprocessor = PreProcessor(chunk_params['max_chunk_len'], chunk_params['overlap_p'])
        return preprocessor

    def __call__(self, ids: Iterable[int], sentences: Iterable[str]) -> torch.Tensor:
        encoded_ids, chunks = self.preprocessor(ids, sentences)
        if len(encoded_ids) != len(chunks):
            logging.info(f"{len(encoded_ids)=}, {len(chunks)=}")
        encoded_input = self.tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.postprocessor(encoded_input, model_output)
        return encoded_ids, sentence_embeddings

