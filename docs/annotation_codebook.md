# Human annotation codebook

**Codebook version:** 1
**Construct ID:** `visual_identity_bias_in_ai_generated_images`
**Construct version:** 1

This codebook is for human annotators. You do not need to read the software
implementation. Automated model labels are **not** shown to you and are **not**
the answer key.

All worked examples below are **fully synthetic / fictional**. They are not
Reddit posts and must not be treated as study data.

---

## Construct

**Name:** Discussion of visual-identity bias in AI-generated images

**Definition:** Whether the post discusses unfair, distorted, or missing
portrayal of **human identity** in **AI-generated images**. In the production
prompt, that portrayal covers identity traits such as race, gender, body type,
disability, age, and culture.

This is a **discourse / perception** construct: it asks what the *post talks
about*. It does **not** ask whether an AI system objectively exhibits bias,
and a `yes` does not prove that any model is biased.

The automated pipeline asks the same yes/no question with a stricter
vocabulary (`yes` / `no` only). Humans may record uncertainty (see labels
below).

---

## Annotation unit

You annotate **one post per task**.

The text you see (`text_to_annotate`) is the post content used for automated
annotation: typically the post **title plus body** after URL/HTML stripping
(pipeline field `clean_text`). Case is preserved. A comment thread is **not**
part of the unit unless that text was already concatenated into the field you
are shown.

You will **not** see:

- the model’s predicted label
- the model’s reasoning or raw output
- the model’s success/failure status
- the original Reddit ID, username, or permalink
- an expected test answer

If the text is empty, `[deleted]`, `[removed]`, or otherwise unusable, use
`insufficient_context` rather than guessing.

---

## Human labels

Use exactly one of:

| Label | Meaning |
|---|---|
| `yes` | The post discusses the construct (see Yes rule). |
| `no` | The post does not discuss the construct (see No rule). |
| `uncertain` | The post is relevant enough to consider, but the construct cannot be resolved confidently. |
| `insufficient_context` | The available text is too incomplete to decide. |

Do **not** force an ambiguous case into `yes` or `no`. The automated model
cannot emit `uncertain`; that limitation is why human labels keep an explicit
uncertainty vocabulary.

Optional `notes` may record a short justification. Notes are private research
material and must not be published with source text.

---

## Yes rule

Label `yes` when identity portrayal in **AI-generated images** is a clear or
implied topic. Operational inclusion — the post does at least one of:

- discusses how AI-generated images misrepresent or exclude identity traits
- mentions racial, gendered, cultural, or bodily **stereotypes in image
  generation**
- critiques underrepresentation or erasure of groups in generated images
- describes image tools failing to depict diverse human appearances

The claim may be the author’s own, or a claim the author is reporting,
endorsing, or disputing, **as long as the post is about that portrayal
topic** (see Edge cases for reported speech and sarcasm).

---

## No rule

Label `no` when the post is **not** about identity portrayal in AI-generated
images. Operational exclusion — for example the post:

- discusses **text-based** AI (chatbots, coding assistants, writing tools)
  without image-identity portrayal
- focuses on technical issues, creativity, or AI-art ethics **unrelated to
  identity portrayal**
- talks about realism, aesthetics, or copyright **without** identity
  representation in images
- mentions censorship, politics, accessibility, or artist livelihoods
  **without** referencing identity representation in images
- mentions identity words only incidentally (see Edge cases)

Generic praise or criticism of an image model is `no` unless identity
portrayal is actually at issue.

---

## Uncertain

Use `uncertain` when the content is on-topic enough to consider, but you
cannot confidently apply `yes` or `no`. Typical situations:

- identity and image generation are both mentioned, but the link is unclear
- the post might be about image quality, lighting, or style rather than
  identity portrayal
- sarcasm or joking tone makes the speaker’s claim unreadable
- a hypothetical is too thin to tell whether the construct is being discussed

Do not use `uncertain` merely because you disagree with the poster.

---

## Insufficient context

Use `insufficient_context` when the **available text** cannot support a
decision, including:

- empty, whitespace-only, `[deleted]`, or `[removed]` text
- truncated fragments with no recoverable topic
- posts that are clearly incomplete (e.g. “title only” with no substance,
  if that is all you are shown)

This is not the same as `uncertain`: here there is not enough material to
judge, rather than enough material that remains ambiguous.

---

## What is outside the construct

Do **not** treat the following as `yes` by themselves:

- any social or political “bias” unrelated to **visual identity in generated
  images**
- evidence (or accusations) that a **system** is objectively biased, unless
  the post actually discusses image-identity portrayal
- keyword presence alone (identity words, “bias”, model names)
- demographic self-description that is not about generated images
- disability / accessibility of **interfaces** (e.g. screen readers) with no
  image-representation claim
- labor, copyright, or “AI art is theft” arguments without identity portrayal

---

## Relationship to keyword filtering

Upstream keyword filters (AI terms, identity/stereotype terms, subreddit
group rules) only decide whether a post **enters** the annotation pool.
Matching a keyword is **not** a `yes`. Many filtered posts should be `no`.

---

## Model prediction vs human label vs reference vs ground truth

| Object | What it is |
|---|---|
| Model prediction | Automated `yes`/`no` after a successful parse |
| Human annotation | Your label, including uncertainty |
| Adjudicated reference | A later resolved `yes`/`no` after disagreement review, if used |
| Scientific ground truth | **Not produced by this codebook.** Even agreed human labels are reliability evidence, not proof that the construct equals “AI bias.” |

Never treat a model prediction as the correct answer.

---

## Edge cases

All examples are **synthetic**.

### Reporting someone else’s claim

If the post is about another person’s claim that generated images
misrepresent identity, that is still discussion of the construct → `yes`,
unless the quote is incidental and the post is about something else.

- **[SYNTHETIC] `yes`:** “A reviewer said fictional tool Pixora only draws
  mayors as one cartoon astronaut; I am trying to reproduce that.”
- **[SYNTHETIC] `no`:** “A reviewer said Pixora is slow; here is my install
  log.”

### Sarcasm

If sarcastic wording still clearly targets identity portrayal in generated
images → `yes`. If you cannot tell → `uncertain`.

- **[SYNTHETIC] `yes`:** “Great, Pixora, another hundred ‘CEOs’ who all look
  like the same man. Super diverse.”
- **[SYNTHETIC] `uncertain`:** “Sure, Pixora is ‘unbiased.’ Anyway, look at
  this lighting.”

### Hypothetical examples

A clearly sketched hypothetical about identity portrayal in generated images
→ `yes`. A one-line what-if with no recoverable topic → `uncertain` or
`insufficient_context`.

- **[SYNTHETIC] `yes`:** “Suppose a generator never produced elders even when
  asked; that would be erasure in the pictures.”

### Generic criticism of AI

- **[SYNTHETIC] `no`:** “LunaDraw is overpriced and the queue is a mess.”
- **[SYNTHETIC] `yes`:** “LunaDraw is a mess: every ‘nurse’ prompt comes back
  as a woman.”

### Technical quality without the construct

- **[SYNTHETIC] `no`:** “Raising sampling steps crashes the fictional CUDA
  driver.”
- **[SYNTHETIC] `no`:** “The skin texture looks waxy; I need a better
  upscaler.” (aesthetics / quality, no identity-portrayal claim)

### Identity terminology appearing incidentally

- **[SYNTHETIC] `no`:** “I am a woman looking for GPU settings for Pixora.”
- **[SYNTHETIC] `yes`:** “Pixora’s ‘woman athlete’ outputs are all the same
  thin body type.”

### Quoted material

Judge the post as a whole. A quote about identity portrayal used to discuss
that topic → `yes`. A quote used only as a joke setup for an unrelated issue
→ `no` or `uncertain`.

### Representation without the target claim

Talk of “representation” in film, hiring, or dataset licensing, with no
AI-generated **image-identity** portrayal → `no`.

- **[SYNTHETIC] `no`:** “We need more representation in our art-school
  faculty.”

### Deleted / removed / incomplete text

- **[SYNTHETIC] `insufficient_context`:** `[removed]`
- **[SYNTHETIC] `insufficient_context`:** empty string

---

## Distinctions you must keep

1. **A post discussing the construct** ≠ **proof that an AI system has the
   construct.** You are labeling discourse.
2. **Your label** ≠ **the model’s prediction.** Do not try to guess the model.
3. **Agreement with another annotator** ≠ **construct validity.**
4. **Keyword hit** ≠ **`yes`.**
