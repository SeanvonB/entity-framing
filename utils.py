import pandas as pd
import spacy
import spacy_experimental

from copy import deepcopy
from datasets import load_dataset

nlp = spacy.load("en_coreference_web_trf")


def collate_documents(path: str, langs: list = ["EN"]) -> None:
    """
    Collate annotations with raw documents and convert to CSV.

    Args:
        path: Path to semeval_train folder with provided file structure
        langs: List of language folders to process

    Returns:
        None
    """
    path = path.rstrip("/")
    header = [
        "document", "mention", "start", "end",
        "superlabel", "label1", "label2", "label3"
    ]

    for lang in langs:
        filename = f"{path}/{lang}/subtask-1-annotations.txt"
        df = pd.read_csv(filename, sep="\t", header=None, names=header)
        texts = []

        for document in df["document"]:
            with open(f"{path}/{lang}/raw-documents/{document}", "r") as f:
                texts.append(f.read())

        df.insert(1, "text", texts, True)
        df.to_csv(f"{path}/subtask1_{lang}_clean.csv", index=False)


def resolve_coref(entity: str, text: str) -> str:
    """
    Resolve all coreferences of entity in provided text.

    Args:
        entity: Entity to resolve coreferences for
        text: Text to resolve coreferences in

    Returns:
        Text with all coreferences of entity resolved
    """
    doc = nlp(text)
    clusters = [v for k, v in doc.spans.items() if k.startswith("coref")]
    mapper = {}
    output = ""

    # Iterate through each coreference Span in SpanGroup
    for cluster in clusters:

        # Skip clusters without entity
        if entity not in [mention.text for mention in cluster]:
            continue

        # Map all other mentions of entity
        for mention in cluster:
            if mention.text == entity:
                continue

            # Replace first token of span with entire entity span
            mapper[mention[0].idx] = entity + mention[0].whitespace_

            # Erase other tokens in span
            for token in mention[1:]:
                mapper[token.idx] = ""

    # Build output by substituting mapper spans for tokens
    for token in doc:
        if token.idx in mapper:
            output += mapper[token.idx]
        else:
            output += token.text + token.whitespace_

    return output
