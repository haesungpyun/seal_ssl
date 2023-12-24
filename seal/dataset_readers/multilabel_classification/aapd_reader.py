from typing import (
    Dict,
    List,
    Union,
    Any,
    Iterator,
    Tuple,
    cast,
    Optional,
    Iterable,
)
import sys
import itertools
if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
import dill
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    ArrayField,
    ListField,
    MetadataField,
    MultiLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token
from .blurb_genre_collection import BlurbGenreReader
import glob

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: MultiLabelField  #: types


@DatasetReader.register("aapd")
class AAPDReader(BlurbGenreReader):

    def example_to_fields(
        self,
        labels: str,
        text:str,
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        
        if meta is None:
            meta = {}

        meta["labels"] = labels
        meta["text"] = text

        x = TextField(
            self._tokenizer.tokenize(text),
        )

        labels = MultiLabelField(list(map(int, list(labels))), skip_indexing=True, num_labels=len(labels))

        return {
            "x": x,
            "labels": labels,
        }

    def text_to_instance(  # type:ignore
        self,
        labels: str,
        text:str,
        **kwargs: Any,
    ) -> Instance:
        
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            labels, text, meta=meta_dict
        )

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )

    