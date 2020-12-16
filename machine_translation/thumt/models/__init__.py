# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.rnnsearch_lrp
import thumt.models.transformer
import thumt.models.transformer_lrp
import thumt.models.transformer_raw_t5
import thumt.models.transformer_raw_t5_nob
import thumt.models.transformer_raw_soft_t5
import thumt.models.transformer_raw_soft_t5_var
import thumt.models.transformer_raw_soft_t5_nob

def get_model(name, lrp=False):
    name = name.lower()

    if name == "rnnsearch":
        if not lrp:
            return thumt.models.rnnsearch.RNNsearch
        else:
            return thumt.models.rnnsearch_lrp.RNNsearchLRP
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        if not lrp:
            return thumt.models.transformer.Transformer
        else:
            return thumt.models.transformer_lrp.TransformerLRP
    elif name == "transformer_raw_t5":
        return thumt.models.transformer_raw_t5.Transformer
    elif name == "transformer_raw_t5_nob":
        return thumt.models.transformer_raw_t5_nob.Transformer
    elif name == "transformer_raw_soft_t5":
        return thumt.models.transformer_raw_soft_t5.Transformer
    elif name == "transformer_raw_soft_t5_var":
        return thumt.models.transformer_raw_soft_t5_var.Transformer
    elif name == "transformer_raw_soft_t5_nob":
        return thumt.models.transformer_raw_soft_t5_nob.Transformer
    else:
        raise LookupError("Unknown model %s" % name)
