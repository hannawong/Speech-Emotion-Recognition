def tensorize_triples(query_tokenizer, queries, bsize): ##transform sentence into ids and masks
    assert bsize is None or len(queries) % bsize == 0
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    batches = []
    for (q_ids, q_mask) in query_batches:
        Q = (q_ids,q_mask)
        batches.append(Q)

    return batches


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
