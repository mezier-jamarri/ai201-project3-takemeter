# TakeMeter Planning Document

## 1. Community
**What community did you choose and why?** 
I am analyzing the `r/ufc` subreddit. `r/ufc` is a highly active, text-heavy space where opinions flow constantly. It is an ideal fit for this classification task because the discourse naturally fractures into distinct categories: those discussing the actual mechanics of the sport, those arguing about fighter legacies, and those just reacting to the chaos of the fights.

## 2. Labels
**What are your 2–4 labels?**

*   **`technical_analysis`**: The post makes a structured argument or observation about fight mechanics, statistics, matchmaking strategy, or grappling/striking technique.
    *   *Example 1:* "Volkanovski's ability to constantly shift stances on the outside completely neutralized Max's jab in their third fight."
    *   *Example 2:* "Merab's takedown chaining is effective not because of his high completion rate, but because of the cardiovascular pace it forces on his opponents."
*   **`hot_take`**: A bold, confident, or sensational claim about a fighter's legacy, mindset, or abilities, stated primarily to spark debate rather than provide evidence.
    *   *Example 1:* "I don't care what anyone says, prime Conor McGregor sleeps current Islam Makhachev inside two rounds."
    *   *Example 2:* "Jon Jones is completely ducking Tom Aspinall and his current heavyweight run is a total joke."
*   **`reaction`**: An immediate, emotional response to a specific fight result, announcement, or press conference moment with little to no actual argument.
    *   *Example 1:* "HOLY SHIT THAT HEAD KICK OUT OF NOWHERE!!!"
    *   *Example 2:* "I am absolutely devastated for Dustin right now, I can't believe it ended like that."

## 3. Hard Edge Cases
**What type of post will be genuinely ambiguous between two labels? How will you handle it?**
*The Ambiguous Post:* "O'Malley is a total fraud because his striking defense is only 61% against elite wrestlers."
This type of post straddles the boundary between `hot_take` (accusatory, emotional framing) and `technical_analysis` (citing a specific metric). 
*Decision Rule:* If the post provides specific, verifiable evidence that breaks down *how* the fighter is failing (e.g., explaining the defensive footwork flaw causing the 61% metric), it will be labeled `technical_analysis`. If the statistic is just cherry-picked window dressing to support an emotional insult or bold claim without mechanical breakdown, it will be labeled `hot_take`.

## 4. Data Collection Plan
**Where will you collect examples? How many per label? What will you do if a label is underrepresented?**
I will manually collect public text posts and high-level comments directly from `r/ufc`. I will gather at least 200 total examples, aiming for a relatively even split (~33% per label). If `technical_analysis` is underrepresented after organic scrolling (which is highly likely on this subreddit), I will specifically search the subreddit using keywords like "breakdown," "technique," or "mechanics" to ensure that class reaches at least a 20% representation threshold in the final dataset.

## 5. Evaluation Metrics
**Which metrics will you use to evaluate your model and why?**
I will evaluate the model using overall accuracy, alongside per-class Precision, Recall, and F1-score. Accuracy alone is insufficient because `r/ufc` discourse heavily skews toward reactions and hot takes; a model could achieve a high accuracy simply by guessing the majority class every time. F1-score is critical to determine if the model is actually learning the nuanced boundary of `technical_analysis` rather than just defaulting to the easiest labels.

## 6. Definition of Success
**What performance would make this classifier genuinely useful?**
To be considered successful, the fine-tuned DistilBERT model must achieve an overall accuracy of >75% and beat the zero-shot Groq baseline by at least 15%. Most importantly, it must achieve an F1-score of >0.70 on the `technical_analysis` label, proving it can reliably separate substantive breakdowns from disguised hot takes.

## 7. AI Tool Plan
*   **Label stress-testing:** Before I begin annotating my 200 examples, I will provide my label definitions and edge case rules to ChatGPT and ask it to generate 10 synthetic posts that sit exactly on the boundary between `hot_take` and `technical_analysis`. If I cannot confidently classify those generated posts using my decision rule, I will tighten my definitions before proceeding with real data.
*   **Annotation assistance:** I will manually label all 200 examples myself to stay close to the data and ensure the boundaries reflect my intended taxonomy. I will not use an LLM for pre-labeling.
*   **Failure analysis:** After evaluating the locked test set, I will paste the list of incorrect predictions into ChatGPT and prompt it to identify semantic or structural patterns in the errors (e.g., "the model consistently misclassifies sarcastic reactions as hot takes"). I will manually verify these patterns by re-reading the examples before writing my final evaluation report.
