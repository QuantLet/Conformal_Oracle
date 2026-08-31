# What stands between this and a submission

Written 2026-08-31. Each item says who it belongs to.

## Verified today

**Elsevier's generative-AI policy**, from
<https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals>.
It asks for two different things and the manuscript previously had only a
partial version of one:

1. A declaration for manuscript preparation, in a prescribed format: *"During
   the preparation of this work, the author(s) used [TOOL] in order to [REASON].
   After using this tool/service, the author(s) reviewed and edited the content
   as needed and take(s) full responsibility for the content of the published
   article."*
2. *"Where AI tools are used as part of the research process rather than
   manuscript preparation, this use should be described in detail in the Methods
   section."* The policy explicitly does not prohibit it: *"This policy does not
   prevent the use of AI tools in formal research design or research methods,
   including but not limited to study design, code development and data
   analysis."*
3. AI may not be listed or cited as an author.

The manuscript's declaration said "language editing and code review", which
understated what was done. Both sections are now written. **The wording is a
statement about the authors' own conduct and needs the authors' approval, not a
build check.** Note that the repository's commit history carries co-authorship
trailers, so the declaration and the replication material must agree.

## Open, and whose

| item | whose | note |
|---|---|---|
| approve or amend the AI declaration and the research-process paragraph | authors | drafted; accuracy is the authors' to affirm |
| upload `conformal-oracle` 0.3.4 | author | built, tested, twine-checked; `cd python && python -m twine upload dist/*`. Section 12 then needs one sentence changed |
| read Zhong (2603.22569) and Cuonzo & Deliu (2606.18199) in full | authors | both are in the bibliography and neither is cited in the text; Cuonzo & Deliu bears on the one-sided novelty claim and has been read from its abstract only |
| decide the final framing of Section 4 | authors | the attribution is written; whether Theorem 4.5 stays stated or is replaced by a citation is a judgement call |
| IRFA word limit, abstract limit, structured-abstract policy, time to first decision | unresolved | the journal's guide-for-authors returns HTTP 403 to a script. Abstract is at 224 words against a commonly cited 250 limit, unverified. Needs a human to open the page |
| whether anyone has documented the Chronos `top_k` truncation defect | unresolved | the research pass returned nothing verified; the manuscript's claim is about one surveyed paper and is stated that narrowly |

## Not blocking

Guards 1--8 pass, the six audits pass, both documents compile and their shipped
PDFs match their sources character for character. 40 of 48 live split sites take
the calibration/test split from one definition; the eight that do not each carry
a reason in `SPLIT_SITES.tsv`.
