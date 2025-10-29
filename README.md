# Know Your Users! Estimating User Domain Knowledge in Conversational Recommenders

<!-- This repository provides resources developed within the following article [[PDF]](): -->


## Summary

The ideal conversational recommender system (CRS) acts like a savvy salesperson, adapting its language and suggestions to each user's level of expertise. However, most current systems treat all users as experts, leading to frustrating and inefficient interactions when users are unfamiliar with a domain. 
Systems that can adapt their conversational strategies to a user's knowledge level stand to offer a much more natural and effective experience. To make a step toward such adaptive systems, we introduce a new task: estimating user domain knowledge from conversations, enabling a CRS to better understand user needs and personalize interactions. A key obstacle to developing such adaptive systems is the lack of suitable data; to our knowledge, no existing dataset captures the conversational behaviors of users with varying levels of domain knowledge. Furthermore, in most dialogue collection protocols, users are free to express their own preferences, which tends to concentrate on popular items and well-known features, offering little insight into how novices explore or learn about unfamiliar features. To address this, we design a game-based data collection protocol that elicits varied expressions of knowledge, release the resulting dataset, and provide an initial analysis to highlight its potential for future work on user-knowledge-aware CRS.

## RecQuest Dataset

Descriptive statistics: number of dialogues (#Dial) and utterances (#Utt) are total counts, while values for the number of turns (#Turns), preferences (#Prefs), and recommendations (#Recs) are reported as mean ± standard deviation. Success rate (Success) is the proportion of dialogues where participants found the target item.

| Domain         | #Dial | #Utt | #Turns (mean ± SD) | #Prefs (mean ± SD) | #Recs (mean ± SD) | Success |
|----------------|-------|------|--------------------|--------------------|-------------------|----------|
| Bicycle        | 79    | 1521 | 9.53 ± 4.71        | 10.00 ± 4.83       | 3.19 ± 1.35       | 60.8%    |
| Digital Camera | 79    | 1687 | 10.35 ± 4.42       | 10.76 ± 5.23       | 3.42 ± 1.72       | 36.7%    |
| Laptop         | 98    | 2179 | 10.00 ± 4.86       | 10.47 ± 4.51       | 3.60 ± 1.27       | 25.5%    |
| Running Shoes  | 179   | 3636 | 10.07 ± 5.37       | 9.90 ± 5.54        | 3.52 ± 1.43       | 21.8%    |
| Smartwatch     | 80    | 1665 | 10.18 ± 3.63       | 9.61 ± 4.08        | 3.45 ± 1.19       | 43.8%    |
| **Total**      | **515** | **10688** | **10.21 ± 4.81** | **10.10 ± 5.03** | **3.46 ± 1.41** | **34.2%** |


<!-- ## Citation

If you use the resources presented in this repository, please cite:

```
@misc{Kostric:2025:arXiv,
    author =          {Ivica Kostric and Ujwal Gadiraju and Krisztian Balog},
    title =           {Know Your Users! Estimating User Domain Knowledge in Conversational Recommenders}, 
    year =            {2025},
    eprint =          {},
    archivePrefix =   {arXiv},
    primaryClass =    {cs.IR},
}  
``` -->

## Contact

Should you have any questions, please contact Ivica Kostric at ivica.kostric[AT]uis.no (with [AT] replaced by @).
