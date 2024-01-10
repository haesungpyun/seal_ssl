import logging
from typing import Dict, List, Iterable, Tuple, Any
from allennlp.data.token_indexers import TokenIndexer
from wcmatch import glob

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp_models.structured_prediction.dataset_readers.srl import SrlReader
from allennlp_models.common.ontonotes import Ontonotes
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


@DatasetReader.register("srl-labeled")
class SrlLabeledReader(SrlReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        bert_model_name: str = None,
        **kwargs,
    ) -> None:
        super().__init__(token_indexers, domain_identifier, bert_model_name, **kwargs)
    
    def _read(self, file_pathes: str):
        # if `file_path` is a URL, redirect to the cache

        for file_path in glob.glob(file_pathes, flags=glob.EXTGLOB):
            file_path = cached_path(file_path)
            ontonotes_reader = Ontonotes()
            logger.info("Reading SRL instances from dataset files at: %s", file_path)
            if self._domain_identifier is not None:
                logger.info(
                    "Filtering to only include file paths containing the %s domain",
                    self._domain_identifier,
                )

            for sentence in self._ontonotes_subset(
                ontonotes_reader, file_path, self._domain_identifier
            ):
                tokens = [Token(t) for t in sentence.words]
                if not sentence.srl_frames:
                    # Sentence contains no predicates.
                    tags = ["O" for _ in tokens]
                    verb_label = [0 for _ in tokens]
                    yield self.text_to_instance(tokens, verb_label, tags)
                else:
                    for (_, tags) in sentence.srl_frames:
                        verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                        yield self.text_to_instance(tokens, verb_indicator, tags)