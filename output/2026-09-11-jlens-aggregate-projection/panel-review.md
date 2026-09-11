# Panel/source admission

Offline preflight ran once as j-lens-aggregate-panel-20260911-MMxZIs,
invocation919d2c7a9d1d4a3fa39f0c8e7e240e7c. Terminal success, MainPID0;
17.405 CPU seconds,746MiB peak host memory, no model or lens loaded.
751 parsed article records:728 eligible,14 short,9 duplicates. The fixed
selection supports448 articles and1792 excerpts without a capacity amendment.

Panel SHA8a034642a4946c4cf7967d28adecb470151101d3de24f10aca8faf5e89bbbe74.
Independent root arithmetic verifies448 article identities, four excerpts each,
no article crossing roles or batches, no overlapping within-article windows,
no exact duplicate token windows, all40 tokens/all-one masks, and exact
1024/256/512 role counts with all sixteen query/two reference groups complete.
Four excerpts per article remain dependent. No population replication claim.

The staged secret scan flagged two JSON values named by tokenizer paths.
They are public-file SHA256 digests, NOT credentials. Independently recomputing
the cached tokenizer.json and tokenizer_config.json digests gives exactly
the frozen panel values (also previously pinned in audited model source).
Only these two exact file/rule/line fingerprints are allowed in repository
.gitleaksignore; full staged secret scanning remains enabled. Panel bytes and
preflight are unchanged, no rerun or scanner bypass.

Acquisition source reviewed fd84bb85e05468d1a68006867d2143c131890dba4b8007fbe09bcc9c41fc3a5c,
worker commit4c52c7eccb0ca97d74d40ad0553fc2aefc7d0304. Root full read plus
eight synthetic tests PASS: exact8-token loss and post-block31 VJP,
no double final norm, parameter freeze, tuple/tensor hooks, cleanup,
strict panel validation, exclusive stage claim and preserved failures.
This is implementation validation, not scientific evidence.
