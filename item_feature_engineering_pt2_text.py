import html
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import spacy
from gensim.models.phrases import Phraser, Phrases

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
RAW_DATA_DIR = Path("data/raw/amazon")
PROCESSED_DATA_DIR = Path("data/processed/amazon")
UNK, TK_REP, TK_WREP = "xxunk", "xxrep", "xxwrep"


class SpacyLemmaTokenizer(object):
    def __init__(self):
        self.tok = spacy.blank("en", disable=["parser", "tagger", "ner"])

    @staticmethod
    def condition(t, min_len=2):
        return not (
            t.is_punct | t.is_space | (t.lemma_ != "-PRON-") | len(t)
            <= min_len | t.is_stop  # | t.is_digit
        )

    def __call__(self, doc):
        return [t.lemma_.lower() for t in self.tok(doc) if self.condition(t)]


class Bigram(object):
    def __init__(self):
        self.phraser = Phraser

    @staticmethod
    def append_bigram(doc, phrases_model):
        doc += [t for t in phrases_model[doc] if "_" in t]
        return doc

    def __call__(self, docs):
        phrases = Phrases(docs, min_count=10)
        bigram = self.phraser(phrases)
        with Pool(cpu_count()) as p:
            docs = p.starmap(self.append_bigram, zip(docs, [bigram] * len(docs)))
        return docs


def spec_add_spaces(t):
    "Add spaces around / and # in `t`. \n"
    return re.sub(r"([/#\n])", r" \1 ", t)


def rm_useless_spaces(t):
    "Remove multiple spaces in `t`."
    return re.sub(" {2,}", " ", t)


def replace_rep(t):
    "Replace repetitions at the character level in `t`."

    def _replace_rep(m):
        c, cc = m.groups()
        return f" {TK_REP} {len(cc)+1} {c} "

    re_rep = re.compile(r"(\S)(\1{3,})")
    return re_rep.sub(_replace_rep, t)


def replace_wrep(t):
    "Replace word repetitions in `t`."

    def _replace_wrep(m):
        c, cc = m.groups()
        return f" {TK_WREP} {len(cc.split())+1} {c} "

    re_wrep = re.compile(r"(\b\w+\W+)(\1{3,})")
    return re_wrep.sub(_replace_wrep, t)


def fix_html(x):
    "List of replacements from html strings in `x`."
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        # .replace("\\n", "\n")
        .replace("\\n", " ")
        .replace("quot;", "'")
        # .replace("<br />", "\n")
        .replace("<br />", " ")
        .replace("</i>", " ")
        .replace("<i>", " ")
        .replace('\\"', '"')
        .replace("<unk>", UNK)
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
        .replace("--", " ")
    )
    return re1.sub(" ", html.unescape(x))


PREPROCESSING_RULES = [
    fix_html,
    replace_rep,
    replace_wrep,
    spec_add_spaces,
    rm_useless_spaces,
]


def preprocessing(t):
    for rule in PREPROCESSING_RULES:
        t = rule(t)
    return t


if __name__ == "__main__":

    meta = pd.read_pickle(PROCESSED_DATA_DIR / "meta_movies_and_tv_processed.p")

    descriptions = meta.description.tolist()
    descriptions_unlisted = []
    for desc in descriptions:
        if len(desc) == 0:
            descriptions_unlisted.append("dempty")
        elif len(desc) > 1:
            descriptions_unlisted.append(" ".join(desc))
        else:
            descriptions_unlisted.append(desc[0])

    with Pool(cpu_count()) as p:
        processed_desc = p.map(preprocessing, [d for d in descriptions_unlisted])

    spacy_tok = SpacyLemmaTokenizer()

    with Pool(cpu_count()) as p:
        tok_descs = p.map(spacy_tok, processed_desc)

    desc_len = [len(d) for d in tok_descs]

    tok_descs_bigr = Bigram()(tok_descs)

    tokenized_descriptions = pd.DataFrame(
        {
            "item": meta.item,
            "description": tok_descs_bigr,
            "description_length": desc_len,
        }
    )
    tokenized_descriptions.to_pickle(PROCESSED_DATA_DIR / "tokenized_descriptions.p")
