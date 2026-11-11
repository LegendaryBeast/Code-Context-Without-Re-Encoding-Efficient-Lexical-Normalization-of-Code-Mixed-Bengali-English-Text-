# Annotation guidelines: adjudicated BanglaMix normalization test subset

Version 1 (before the pilot). Revise after the pilot; record every change in the change log at the end.

## 1. Goal

BanglaMix gives one silver Bengali-script reference per comment. Your task is to **check that reference token by
token, correct its errors, and record other spellings that are equally correct.** The result is an adjudicated
reference for 600 test comments (all 464 code-mixed comments of the split-42 test set and 136 monolingual ones).

Three rules come before everything else:

1. **Work independently.** Do not discuss items with the other annotator until both sheets are submitted.
2. **Never look at system outputs.** The sheet contains no model output; do not open prediction files while annotating.
3. **Follow the corpus conventions, fix only errors.** The systems were trained on BanglaMix. A reference must not be
   changed merely because you would have written it differently; change it only when it is wrong. When the silver
   form is one acceptable form among several, keep it and add the others as variants.

## 2. The sheet

One row per source token. Grey columns are information; yellow columns are yours.

| Column | Meaning |
|---|---|
| `item_id`, `comment_no`, `token_no` | identifiers; do not edit |
| `source_token`, `lid_tag` | the token as written and its corpus language tag |
| `context` | the whole comment, with this token marked ⟦like this⟧ |
| `silver_comment_reference` | the whole silver reference (first row of each comment only) |
| `silver_target` | the part of the silver reference that belongs to this token |
| **`corrected_target`** | prefilled with `silver_target`. Keep it if correct; otherwise overwrite it |
| **`acceptable_variants`** | other equally correct forms, separated by `\|`; empty if none |
| **`flag`** | only when needed: `source unclear`, `token misaligned`, `offensive or personal data`, `other (see notes)` |
| **`notes`** | short explanation; required with `other (see notes)` |

Special forms in targets:

- **Several words.** One source token can map to several target words, separated by spaces:
  `Bhaiনাকি` → `ভাই নাকি`.
- **`~` (glue).** A target starting with `~` attaches to the previous word without a space. It appears where the
  reference joins two source tokens (`now` `.` → `নাও` `~।`) or merges pieces of one word.
- **Empty.** Leave `corrected_target` empty only when the token produces no output of its own because its text is
  already part of a neighbouring target.

For the 63 comments whose reference is not token-aligned in the corpus, the split into `silver_target`s was made
automatically. If the words of the reference are correct but assigned to the wrong source token, move them to the
right rows and set `flag = token misaligned`.

## 3. Decisions for each token

Ask, in order:

1. **Is `silver_target` a correct Bengali-script rendering of this token in this comment?** If yes, keep it.
2. **If not, what is the correct form?** Write it in `corrected_target`, following Section 4.
3. **Are other forms equally correct here?** Add them to `acceptable_variants`. A variant must be a form a careful
   writer of colloquial Bengali would accept as correct for this word in this context. It is not a list of
   plausible typos.

Read the whole comment first. Several Romanized words are homographs whose Bengali form depends on context:

| Source | Possible targets | Decided by |
|---|---|---|
| `ar` | আর ("and", "more") / এর (genitive) | the following word |
| `ase` | আছে ("is, has") / আসে ("comes") | the sentence |
| `ki` | কি / কী | question vs. pronoun; both are often acceptable, so list the other as a variant |

## 4. Conventions, with examples from BanglaMix

**Bengali-script words (BAN).** Keep them unless they are misspelled; 94% of BAN tokens are copied unchanged in the
corpus. Correct spelling errors (`মুটামুটি` → `মোটামুটি`, `চরিএ` → `চরিত্র`, `বিডিও` → `ভিডিও`). Do **not**
standardize colloquial or dialect grammar: `করতেছি` stays `করতেছি`.

**Romanized Bengali (ROM).** Transliterate the word the writer intended, keeping its colloquial form:
`kortaci` → `করতেছি`, `ami` → `আমি`, `ekjon` → `একজন`. Mark spelling variants of names and of words with free vowel
length as variants (`Rezvy`: `রিজভি` | `রেজভী`; `Etto`: `এত্তো` | `এত`).

**English words (ENG).** Write them phonetically in Bengali script, as the corpus does; never translate them:
`please` → `প্লিজ`, `video` → `ভিডিও`, `thanks` → `থ্যাঙ্কস`, `phone` → `ফোন`, `bro` → `ব্রো`,
`subscribe` → `সাবস্ক্রাইব`. Proper nouns and brands are adapted the same way: `Bangladesh` → `বাংলাদেশ`,
`iPhone` → `আইফোন`, `YouTube` → `ইউটিউব`. A target left in Latin letters (`Please`, `squad`) is an error unless the
token is a URL, a user handle, a code, or an unreadable string (`yanim100000` stays as written). Common alternative
spellings are variants (`tour`: `ট্যুর` | `টুর`; `Ducky`: `ডাকি` | `ডাকী`).

**Emoji and symbols (OTH).** Copy them unchanged, including when they are glued to a word: `nai😢` → `নাই😢`,
`people❤❤❤` → `পিপল❤❤❤`. Keep the number of repeated emoji and punctuation marks as written.

**Punctuation.** Keep marks as written. A full stop may appear as `.` or `।` in the corpus; both are acceptable, so
keep the silver form and add the other as a variant (`Instagram.`: `ইনস্টাগ্রাম.` | `ইনস্টাগ্রাম।`).

**Digits.** The corpus writes numbers both with ASCII digits (`2012`, `17`) and with Bengali digits (`১`). Both are
acceptable: keep the silver form and add the other digit form as a variant (`20` → `20` | `২০`). Numbers with letters
(`20k`) follow the same rule for their digits.

**Elongation.** Keep the writer's elongation where the reference keeps it (`MEEEEEEE` → `মিইইইইইই`); a different
number of repeated letters is a variant, not an error.

**Glued tokens (UNK).** Split or join exactly as the meaning requires, using several words or `~` (Section 2):
`plz..` → `প্লিজ..`, `korenna....music` → `করেন না....মিউজিক` | `করেননা....মিউজিক`.

**Wrong word.** If the silver target is a different word altogether (`mera` → `দাও`, `mone` → `থাকে`), it is an
error; write the correct form.

## 5. Flags

- `source unclear`: the source token cannot be read with confidence. Still give your best `corrected_target`.
- `token misaligned`: the reference words are right but sit on the wrong rows (Section 2); fix the rows.
- `offensive or personal data`: the comment contains personal data (phone numbers, addresses) or abusive content.
  Annotate it normally; the flag is for reporting only.
- `other (see notes)`: anything else; explain in `notes`.

## 6. Procedure

1. **Pilot.** Both annotators label `sheets/pilot_annotator_A.xlsx` and `..._B.xlsx` (the first 50 comments,
   567 tokens) independently.
2. **Agreement check.** Run `python annotation/adjudicated_test/agreement.py --a <sheet A> --b <sheet B>`. Discuss
   only the disagreements it lists, refine these guidelines, and record the changes below.
3. **Full annotation.** Both annotators label the full sheets (600 comments, 7,275 tokens) independently. The pilot
   comments come first; re-check them under the revised guidelines.
4. **Adjudication.** `agreement.py` writes an adjudication sheet with every token on which the annotators differ.
   The adjudicator chooses the final target and the final variants for those tokens only. Tokens on which both
   annotators agree are final automatically.
5. **Scoring.** `python annotation/adjudicated_test/evaluate.py --final <adjudicated sheet>` scores every
   controlled system against the silver and the adjudicated references.

Save sheets as `.xlsx` with a new file name (for example `full_annotator_A_done.xlsx`). Do not reorder rows, rename
sheets, or edit grey columns; the scripts read `item_id`, `corrected_target`, `acceptable_variants`, `flag`, and
`notes` only.

## Change log

- Version 1: initial version.
