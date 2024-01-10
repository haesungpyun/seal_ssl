import logging
from typing import Dict, List, Iterable, Tuple, Any
from wcmatch import glob

from transformers.models.bert.tokenization_bert import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp_models.common.ontonotes import Ontonotes, OntonotesSentence
from allennlp_models.structured_prediction.dataset_readers.srl import (
    SrlReader,
    _convert_tags_to_wordpiece_tags,
    _convert_verb_indices_to_wordpiece_indices
)

logger = logging.getLogger(__name__)


@DatasetReader.register("srl-unlabeled")
class SrlUnlabeledReader(SrlReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        bert_model_name: str = None,
        max_length: int = None,        
        use_pos: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(token_indexers, domain_identifier, bert_model_name, **kwargs)
               
        self._max_length = max_length
        self.use_pos = use_pos

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
                # if len(tokens) > self._max_length:
                #     continue  
                if not sentence.srl_frames:
                    # Sentence contains no predicates.
                    tags = ["O" for _ in tokens]
                    verb_label = [0 for _ in tokens]
                
                    yield self.text_to_instance(tokens, verb_label, tags)
                else:
                    if self.use_pos:                
                        from nltk import pos_tag
                        word_pos_tags = pos_tag(sentence.words)
                        tags = [tag for (_, tag) in word_pos_tags]
                        sentence.srl_frames = list(filter(lambda x: (x[0], None) if 'V' in x[1] else None, word_pos_tags))

                        for (verb, _) in sentence.srl_frames:
                            verb_indicator = [1 if word == verb else 0 for word in sentence.words]
                            yield self.text_to_instance(tokens, verb_indicator, tags)
                    else:        
                        for (_, tags) in sentence.srl_frames:
                            verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                            yield self.text_to_instance(tokens, verb_indicator, tags)

    def text_to_instance(  # type: ignore
        self, tokens: List[Token], verb_label: List[int], tags: List[str] = None,
        mask_tag: bool = True
    ) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """

        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(
                [t.text for t in tokens]
            )
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField(
                [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                token_indexers=self._token_indexers,
            )
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields["tokens"] = text_field
        fields["verb_indicator"] = verb_indicator

        if all(x == 0 for x in verb_label):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index
        metadata_dict['data_type'] = 'unlabeled'
        
        if tags:
            if mask_tag:
                tags = ['O' for _ in range(offsets[-1])]
                unlabel_tags = [-100 for _ in range(offsets[-1]+2)]
            else:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                unlabel_tags = new_tags
        metadata_dict["gold_tags"] = tags
        fields["tags"] = SequenceLabelField(unlabel_tags, text_field)
        fields["metadata"] = MetadataField(metadata_dict)
        
        return Instance(fields)
