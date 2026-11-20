# TakeMeter — `r/ufc` Discourse Classifier

**Demo Video:**(https://drive.google.com/file/d/1cRQJX1iSLo_BsSkyXMLLkeZnXva-pgVl/view?usp=sharing)  
**Labeled Dataset:** [`ufc_dataset.csv`](./ufc_dataset.csv)  
**Planning Spec:** [`planning.md`](./planning.md)  
**Evaluation Outputs:** [`evaluation_results.json`](./evaluation_results.json) | [`confusion_matrix.png`](./confusion_matrix.png)

TakeMeter is a fine-tuned `distilbert-base-uncased` text classifier that evaluates discourse quality in the `r/ufc` subreddit by distinguishing substantive technical fight breakdowns from sensational hot takes and immediate emotional reactions.

---

## 1. Community Choice & Reasoning

I chose **`r/ufc`**, one of the largest and most active MMA discussion communities on Reddit. Online MMA discourse is notoriously polarized between fans who break down fight mechanics (footwork, distance management, grappling transitions, cardio pacing) and fans who engage in hyperbolic legacy debates or live-event hype. Because `r/ufc` is text-heavy and spans everything from live fight threads to technique breakdowns, it provides a realistic, noisy classification challenge where separating signal (`technical_analysis`) from noise (`hot_take` and `reaction`) is genuinely useful to community members.

---

## 2. Label Taxonomy

The taxonomy consists of three mutually exclusive labels grounded in `r/ufc` community norms:

* **`technical_analysis`**: The post makes a structured argument or observation about fight mechanics, statistics, matchmaking strategy, or grappling/striking technique.
  * *Example 1:* "Volkanovski's ability to constantly shift stances on the outside completely neutralized Max's jab in their third fight."
  * *Example 2:* "Merab's takedown chaining is effective not because of his high completion rate, but because of the cardiovascular pace it forces on his opponents."
* **`hot_take`**: A bold, confident, or sensational claim about a fighter's legacy, mindset, or abilities, stated primarily to spark debate rather than provide mechanical evidence.
  * *Example 1:* "I don't care what anyone says, prime Conor McGregor sleeps current Islam Makhachev inside two rounds."
  * *Example 2:* "Jon Jones is completely ducking Tom Aspinall and his current heavyweight run is a total joke."
* **`reaction`**: An immediate, emotional response to a specific fight result, announcement, or press conference moment with little to no actual argument.
  * *Example 1:* "HOLY SHIT THAT HEAD KICK OUT OF NOWHERE!!!"
  * *Example 2:* "I am absolutely devastated for Dustin right now, I can't believe it ended like that."

**Edge Case Decision Rule:** If a post cites a statistic or technical detail solely as window dressing to support an emotional insult or bold claim without explaining *how* the fight mechanics work, it is labeled `hot_take`. If it explains the physical mechanic or tactical cause behind the outcome, it is labeled `technical_analysis`.

---

## 3. Dataset Collection & Annotation

### Source & Labeling Process
All **206 examples** were collected from public `r/ufc` threads exported as plain text, spanning:
* Live event discussion threads (e.g., UFC 334 Live Discussion Thread)
* Rankings, "Fight of the Year," and "Most Well-Rounded Fighter" debate threads
* Technique and fight-style breakdown threads (e.g., early MMA metas, striking mechanics, and style matchups)

Low-effort noise (GIFs, single-word replies, off-topic banter, and duplicates) was stripped out. Initial candidate extraction and pre-labeling were assisted by an LLM using my `planning.md` definitions, followed by a strict manual audit where stretched or borderline live-thread comments were removed and replaced with cleaner examples from targeted technique and debate threads.

### Label Distribution (`ufc_dataset.csv`)
| Label | Count | Share |
| :--- | :---: | :---: |
| `technical_analysis` | 71 | 34.5% |
| `hot_take` | 70 | 34.0% |
| `reaction` | 65 | 31.5% |
| **Total** | **206** | **100.0%** |

No class exceeds 35% of the dataset, avoiding majority-class bias.

### Three Difficult-to-Label Examples
1. **Post:** *"IMO Gane still hasn’t defending his title or even earned it. Every single title fight he’s had has had some sort of illegal bullshit going on. Eye fucked Aspinall in both eyes, Back of the head shots to Pereira, Eye poked Hokit. Until he defends the belt without doing an illegal move he’s the disputed champion"*
   * **Ambiguity:** Lists three specific historical fight events like an argument (`technical_analysis`), but the core claim is an accusatory, debate-provoking legacy statement (`hot_take`).
   * **Decision:** Labeled **`hot_take`**. Following the edge-case rule, the cited fouls are used as ammunition for a provocative claim ("disputed champion") rather than a mechanical breakdown of fighting technique.
2. **Post:** *"While Alex Pereira does have a good record on paper, you simply can’t deny the fact that he’s has very favourable matchups against one dimensional fighters with no wrestling. And the general consensus right now is that the MW champion could go up and rag-doll Alex at LHW. You can’t rank him above Topuria when that’s the general sentiment."*
   * **Ambiguity:** References stylistic matchups and wrestling vulnerabilities (`technical_analysis`), but frames them around P4P ranking debates and hypotheticals (`hot_take`).
   * **Decision:** Labeled **`hot_take`**. It asserts a ranking argument using broad stylistic generalizations rather than breaking down specific grappling or striking mechanics.
3. **Post:** *"Rogan was so bias in that fight in favour of Hokit, according to Joe it's okay to be a punchbag as long as you are applying pressure"*
   * **Ambiguity:** Mentions pressure vs. striking defense (`technical_analysis`), but is primarily venting about live commentary (`reaction`).
   * **Decision:** Labeled **`reaction`**. The comment is an immediate emotional complaint about the broadcast during a live fight rather than a structured tactical breakdown.

---

## 4. Fine-Tuning Approach

* **Base Model:** `distilbert-base-uncased` (HuggingFace `AutoModelForSequenceClassification` with a 3-class classification head).
* **Data Split:** Stratified 70% / 15% / 15% split (`random_state=42`): **144 train** (50 / 49 / 45), **31 validation**, and **31 test** (11 `technical_analysis`, 10 `hot_take`, 10 `reaction`).
* **Training Setup:** Google Colab T4 GPU, `max_length=256`, `per_device_train_batch_size=16`, `weight_decay=0.01`, `load_best_model_at_end=True` (selected by validation accuracy).

### Key Hyperparameter Decision
The starter notebook's default settings (`num_train_epochs=3`, `learning_rate=2e-5`, `warmup_steps=50`) caused severe underfitting on 144 training examples: with batch size 16, 3 epochs is only **27 total optimizer steps**, meaning `warmup_steps=50` prevented the learning rate from ever reaching its target. The first run stalled at `~1.098` loss ($\ln 3$, random guessing), predicted `reaction` 0 times, and scored **0.387** test accuracy.

I fixed this by adjusting three hyperparameters:
* `warmup_steps`: `50` → **`5`** (~10% of total steps)
* `num_train_epochs`: `3` → **`6`** (54 total steps)
* `learning_rate`: `2e-5` → **`3e-5`**

With these changes, validation accuracy climbed steadily to **0.871** by epoch 6, and locked test accuracy jumped from **0.387** to **0.774**.

---

## 5. Zero-Shot Baseline (Groq)

The baseline used **`openai/gpt-oss-120b`** via the Groq API (`temperature=0`, no `max_tokens` cap so the reasoning model could finish its output cleanly) on the exact same 31-example locked test set, with a `0.1s` delay between requests. All **31/31** responses were cleanly parseable.

**System Prompt Used:**
```text
You are classifying posts and comments from the r/ufc Reddit community.
Assign each post to exactly one of the following categories.

technical_analysis: The post makes a structured argument or observation about fight mechanics, statistics, matchmaking strategy, or grappling/striking technique.
Example: "Volkanovski's ability to constantly shift stances on the outside completely neutralized Max's jab in their third fight."

hot_take: A bold, confident, or sensational claim about a fighter's legacy, mindset, or abilities, stated primarily to spark debate rather than provide mechanical evidence.
Example: "I don't care what anyone says, prime Conor McGregor sleeps current Islam Makhachev inside two rounds."

reaction: An immediate, emotional response to a specific fight result, announcement, or press conference moment with little to no actual argument.
Example: "HOLY SHIT THAT HEAD KICK OUT OF NOWHERE!!!"

Decision rule for borderline posts: If a post cites a stat or technical detail only as window dressing to support an emotional insult or bold claim without explaining the fight mechanics, classify it as hot_take.

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
technical_analysis
hot_take
reaction
```
## 6. Evaluation Report

### Overall Comparison (Locked Test Set, n = 31)
| Model | Overall Accuracy | Macro F1 | Delta vs. Baseline |
| :--- | :---: | :---: | :---: |
| Zero-shot baseline (`openai/gpt-oss-120b` via Groq) | **0.710** | 0.67 | — |
| Fine-tuned DistilBERT (`distilbert-base-uncased`) | **0.774** | **0.78** | **+0.065 (+6.5%)** |

### Per-Class Metrics

**Zero-Shot Baseline (Groq):**
| Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `technical_analysis` | 1.00 | 0.27 | 0.43 | 11 |
| `hot_take` | 0.60 | 0.90 | 0.72 | 10 |
| `reaction` | 0.77 | 1.00 | 0.87 | 10 |
| **Macro Avg** | **0.79** | **0.72** | **0.67** | **31** |

**Fine-Tuned DistilBERT:**
| Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `technical_analysis` | 0.69 | 0.82 | 0.75 | 11 |
| `hot_take` | 0.70 | 0.70 | 0.70 | 10 |
| `reaction` | 1.00 | 0.80 | 0.89 | 10 |
| **Macro Avg** | **0.80** | **0.77** | **0.78** | **31** |

### Confusion Matrix — Fine-Tuned DistilBERT (Test Set)
Rows are **True Labels**; columns are **Predicted Labels**. (Supplementary visual: [`confusion_matrix.png`](./confusion_matrix.png))

| True \ Predicted | `technical_analysis` | `hot_take` | `reaction` |
| :--- | :---: | :---: | :---: |
| **`technical_analysis`** | **9** | 2 | 0 |
| **`hot_take`** | 3 | **7** | 0 |
| **`reaction`** | 1 | 1 | **8** |

### Sample Classifications (Fine-Tuned DistilBERT)
| Post (truncated) | True Label | Predicted | Confidence | Correct? |
| :--- | :--- | :--- | :---: | :---: |
| I know I'll be raked over the coals, but FOTY is Hokit vs Gane. We haven't seen a HW w tha... | `hot_take` | `hot_take` | 0.50 | yes |
| The UFC has always had a lack luster HW division with small spurts of greatness. Its been ... | `hot_take` | `hot_take` | 0.52 | yes |
| McKinney is very capable of submissions. But also very capable of getting subbed especiall... | `technical_analysis` | `technical_analysis` | 0.52 | yes |
| He is a failure of a fighter who is meeting ends meet by doing a terrible gimmick that doe... | `hot_take` | `technical_analysis` | 0.51 | no |
| While Alex Pereira does have a good record on paper, you simply can’t deny the fact that h... | `hot_take` | `technical_analysis` | 0.62 | no |

**Why the correct prediction is reasonable:** In row 3 (*"McKinney is very capable of submissions. But also very capable of getting subbed especially RNC. His cardio def hasn't improved..."*), the model correctly predicted `technical_analysis` (0.52 confidence) because the post evaluates a fighter's specific grappling vulnerabilities (rear-naked choke) and cardiovascular conditioning across a multi-fight sample rather than making an emotional or legacy-based hot take.

### Error Analysis (3 Specific Wrong Predictions)
Five of the seven total errors (71%) occurred strictly between `hot_take` and `technical_analysis`, while `reaction` achieved **1.00 precision** (zero false positives).

1. **Error #2 — `hot_take` misclassified as `technical_analysis` (confidence: 0.62)**
   * **Text:** *"While Alex Pereira does have a good record on paper, you simply can’t deny the fact that he’s has very favourable matchups against one dimensional fighters with no wrestling. And the general consensus..."*
   * **Why it failed:** This post is long, structured, and uses MMA vocabulary ("record on paper," "favourable matchups," "one dimensional fighters," "wrestling"). DistilBERT latched onto post length and domain vocabulary as proxies for `technical_analysis`, missing that the underlying intent is a subjective P4P ranking argument rather than a mechanical breakdown.
   * **How to fix:** Add more multi-sentence `hot_take` examples that use analytical-sounding vocabulary (e.g., matchmaking, wrestling, records) to make bold ranking claims so the model learns to separate *tone/vocabulary* from *mechanical explanation*.

2. **Error #3 — `technical_analysis` misclassified as `hot_take` (confidence: 0.55)**
   * **Text:** *"Radtke looking like shit. He's definitely reading too much cause Wonderboy is faster, longer, and more accurate than he expected."*
   * **Why it failed:** The opening clause (*"Radtke looking like shit"*) uses blunt, insulting slang typical of `hot_take` posts. Even though the second sentence explains the tactical cause (speed, reach, and accuracy forcing hesitation), the model over-weighted the aggressive opening phrase.
   * **How to fix:** Include more `technical_analysis` examples written in casual, profane Reddit prose so the classifier does not equate profanity or blunt phrasing with `hot_take`.

3. **Error #7 — `reaction` misclassified as `technical_analysis` (confidence: 0.65)**
   * **Text:** *"Rogan was so bias in that fight in favour of Hokit, according to Joe it's okay to be a punchbag as long as you are applying pressure"*
   * **Why it failed:** The post is a live-thread complaint about commentary (`reaction`), but its second clause mentions "applying pressure" and getting hit ("punchbag"). Because "pressure" appears frequently in `technical_analysis` training rows, the model misrouted a broadcast rant into `technical_analysis`.
   * **How to fix:** Add more `reaction` examples that reference in-fight moments or commentary using MMA terms (e.g., "pressure," "takedowns," "eye pokes") while remaining purely emotional complaints.

---

## 7. Model Reflection: Learned vs. Intended

I intended the model to learn **discourse intent**: whether a commenter is explaining *how* a fight works (`technical_analysis`), making a provocative claim to spark debate (`hot_take`), or venting emotionally in the moment (`reaction`).

In practice, DistilBERT learned a strong separation between `reaction` (short, exclamatory, all-caps or emotional phrasing — 0.89 F1) and the other two classes, and it dramatically outperformed the zero-shot LLM on `technical_analysis` recall (0.82 vs. 0.27). However, on the boundary between `hot_take` and `technical_analysis`, the model partially relied on **surface shortcuts**: longer posts with MMA terminology ("wrestling," "matchups," "pressure") get pulled toward `technical_analysis`, while shorter or more profane posts get pulled toward `hot_take`. With only 144 training examples, the model captured vocabulary and length correlations well, but still struggles when a `hot_take` borrows analytical language or a `technical_analysis` uses blunt Reddit slang.

---

## 8. Spec Reflection

* **How `planning.md` helped:** Defining the explicit **Hard Edge Case rule** and setting a **per-class F1 > 0.70 threshold for `technical_analysis`** before training kept the project honest. When the zero-shot baseline hit 0.710 overall accuracy, the per-class F1 metric immediately exposed that it was actually failing on `technical_analysis` (0.43 F1, 0.27 recall). Likewise, when the first DistilBERT run scored 0.387, my spec criteria forced me to diagnose the `warmup_steps=50` bug rather than settling for a broken model.
* **How implementation diverged from `planning.md`:** Two things diverged. First, I originally planned to manually label all 200 examples without LLM assistance, but scraping raw live threads produced hundreds of noisy comments; I pivoted to using an LLM to parse and pre-label raw thread dumps, then strictly audited and pruned them myself. Second, I set a success threshold of beating the baseline by +15%—in reality, the 120B-parameter Groq baseline was stronger than expected (0.710), so my fine-tuned DistilBERT beat it by **+6.5%** (0.774), while still meeting both my >75% overall accuracy goal and my >0.70 `technical_analysis` F1 goal (0.75).

---

## 9. AI Usage Disclosure

I used an LLM (Gemini) as a collaborator in three specific parts of the workflow:

1. **Dataset Parsing & Pre-Labeling (Disclosed Annotation Assistance):** I exported raw `r/ufc` threads as `.txt` files and directed the AI to strip metadata/GIFs and pre-label comments using my `planning.md` taxonomy. On the second batch, the AI stretched borderline live-thread comments to force an artificial 70/70/70 split. I caught this, overrode the batch, and directed a strict re-audit that dropped 55 stretched rows—then I pulled additional targeted technique and debate threads to reach 206 clean, unforced examples.
2. **Hyperparameter Debugging:** When my initial fine-tuning run produced 0.387 accuracy and 0.00 F1 on `reaction`, I shared the evaluation output and confusion matrix with the AI. It identified that `warmup_steps=50` exceeded my 27 total training steps (144 examples / batch size 16 × 3 epochs). I verified the step math, updated `warmup_steps=5`, `num_train_epochs=6`, and `learning_rate=3e-5`, and re-ran training, which raised test accuracy to 0.774.
3. **Failure Pattern Analysis:** After evaluating the locked test set, I shared the 7 misclassified examples with the AI to surface recurring error themes (vocabulary/length shortcuts between `hot_take` and `technical_analysis`), which I manually verified against the raw test rows before writing the Error Analysis section.
