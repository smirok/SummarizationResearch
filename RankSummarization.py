from summa.pagerank_weighted import pagerank_weighted_scipy as pagerank
from nltk import sent_tokenize
from summa.commons import build_graph, remove_unreachable_nodes
from summa.preprocessing.textcleaner import clean_text_by_sentences, init_textcleanner, split_sentences, filter_words, \
    merge_syntactic_units
from summa.summarizer import _set_graph_edge_weights, _add_scores_to_sentences, _extract_most_important_sentences


def real_clean_text_by_sentences(text, language="english", additional_stopwords=None):
    init_textcleanner(language, additional_stopwords)
    original_sentences = sent_tokenize(text)
    filtered_sentences = filter_words(original_sentences)

    return merge_syntactic_units(original_sentences, filtered_sentences)


def rank_summarize(text, ratio=0.2, words=None, language="english", split=False, scores=False,
                   additional_stopwords=None):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    # Gets a list of processed sentences.
    sentences = real_clean_text_by_sentences(text)

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    # Remove all nodes with all edges weights equal to zero.
    remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return [] if split else ""

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = pagerank(graph)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_most_important_sentences(sentences, ratio, words)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return list(map(lambda sentence: sentence.score, extracted_sentences))
