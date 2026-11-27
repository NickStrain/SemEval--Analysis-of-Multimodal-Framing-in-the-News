---
license: mit
task_categories:
- text-generation
- text-classification
language:
- en
pretty_name: Multi-Modal Framing Analysis
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: full
    path: annotated_data.csv
  - split: valid_framing_subset
    path: framing_subset.csv
---

# Dataset Card for MM Framing

arXiv: https://arxiv.org/abs/2503.20960

## Dataset Details

### Dataset Description

### Dataset Sources
        
### Column name descriptions:

- *uuid* - Unique ID for each article
- *title* - Title of the article
- *date_publish* - Publication date
- *source_domain* - Domain of the publisher
- *url* - Article URL
- *political_leaning* - Political leaning of the publisher
  
---------- Annotations ---------------
- *text-topic* - Article topic generated from article text
- *text-topic-exp* - Article topic explanation
- *text-entity-name* - Main entity in article text
- *text-entity-sentiment* - Sentiment towards main entity
- *text-entity-sentiment-exp* - Explanation of text sentiment
- *text-generic-frame* - Generic Frame used in Article text
- *text-generic-frame-exp* - Generic Frame in text explanation
- *text-issue-frame* - Issue Frame used in article text
- *text-issue-frame-exp* - Issue Frame explanation
- *img-generic-frame* - Generic Frame used in Article Image
- *img-frame-exp* - Generic Frame in image explanation
- *img-entity-name* - Main subject in Article Image
- *img-entity-sentiment* - Sentiment towards the subject in Article image
- *img-entity-sentiment-exp* - Explanation of image sentiment
- *gpt-topic* - Consolidated topic

## Uses

The dataset was created to support media framing analysis.

### Direct Use

This dataset can be used for analysis of frames communicated in news media.

### Recommendations

The dataset is for analysis and reasearch purposes only.

## Citation

If you find our dataset helpful, kindly refer to us in your work using the following citation:
```
@misc{arora2025multimodalframinganalysisnews,
      title={Multi-Modal Framing Analysis of News}, 
      author={Arnav Arora and Srishti Yadav and Maria Antoniak and Serge Belongie and Isabelle Augenstein},
      year={2025},
      eprint={2503.20960},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.20960}, 
}
```