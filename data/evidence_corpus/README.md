# Evidence corpus for M3 retrieval-augmented prompting

Short, paraphrased clinical-practice statements that the M3 model retrieves
from at inference time. We do **not** reproduce ADA-copyrighted material;
each card paraphrases standard practice from openly available specialty
society guidelines (AAE, AAP, AAOMS) and the dental clinical literature.

Each `.md` file is one *evidence card*: a short paragraph (3–5 sentences)
on a single decision rule. Cards are numbered for stable citation
(`E001`, `E002`, ...) so the model can refer to a specific card in its
rationale.

These cards are intentionally minimal: the goal is to give the LLM a
focused signal about *standard* practice for the kind of decision it is
making, not to encode all of dentistry.

Citation format used by retrievals: `[E###]` inserted inline by the model.
