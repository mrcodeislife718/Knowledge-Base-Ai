# Stage-B certainty questions

Before Knowledge-Base-AI may claim technical superiority over a named competitor, the proof system must be able to answer all of the following with evidence rather than assumption.

1. Are we comparing the exact official upstream implementations and pinned commits?
2. Are all systems using the same corpus, query set, answer targets, model family, embedding family, reranker policy, context budget, token budget, hardware/API envelope, concurrency, cache policy and retry policy?
3. Are indexes built from scratch under equivalent conditions, and are indexing time, indexing cost, index size and query-time cost all measured separately?
4. Are held-out questions genuinely hidden from implementation tuning, and is benchmark contamination/leakage checked?
5. Are easy, hard, adversarial, temporal, contradiction, multi-hop, source-independence and poisoned-input cases all represented?
6. Is source-independence measured by origin/ancestry rather than URL count so syndication cannot inflate evidence?
7. Are we measuring answer correctness, evidence recall, evidence precision, citation faithfulness, contradiction handling, calibration, abstention quality and failure severity—not just retrieval recall?
8. Are repeated trials randomized, paired by case, powered adequately, corrected for multiple comparisons and reported with confidence intervals/effect sizes?
9. Are results stable across seeds, corpora, domains, corpus sizes, context sizes and hardware/provider environments?
10. Do ablations prove which Knowledge-Base-AI mechanisms actually create the gain?
11. Do negative controls detect a broken or biased benchmark that would make every system appear good?
12. Can an independent reviewer reproduce the run from a frozen manifest containing code SHAs, environment, datasets, configs and random seeds?
13. Are failures and losing categories preserved rather than hidden, and does the claim language narrow itself to exactly what was proved?
14. Would the claim remain true after excluding our best dataset, our best seed, or our best metric?
15. Does the system refuse to issue a superiority claim when any required competitor failed, used a proxy, differed materially in resources, or produced incomplete evidence?

The implementation treats these questions as gates, not documentation advice.