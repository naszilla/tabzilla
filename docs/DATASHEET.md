# Datasheet for TabZilla

We include a [Datasheet](https://arxiv.org/abs/1803.09010). 
Thanks for the Markdown template from [Christian Garbin's repository](https://github.com/fau-masters-collected-works-cgarbin/datasheet-for-dataset-template).

Jump to section:

- [Motivation](#motivation)
- [Dataset Composition](#dataset-composition)
- [Collection Process](#collection-process)
- [Data Preprocessing](#data-preprocessing)
- [Data Distribution](#data-distribution)
- [Dataset Maintenance](#dataset-maintenance)
- [Legal and Ethical Considerations](#legal-and-ethical-considerations)

## Motivation

### Why was the datasheet created? (e.g., was there a specific task in mind? was there a specific gap that needed to be filled?)

The goal of releasing the TabZilla Benchmark Suite is to accelerate research in tabular data by introducing a set of 'hard' datasets. Specifically, simple baselines cannot reach top performance, and most algorithms (out of the 19 we tried) cannot reach top performance. We found that a surprisingly high percentage of datasets used in tabular research today are such that a simple baseline can reach just as high accuracy as the leading methods.

### Has the dataset been used already? If so, where are the results so others can compare (e.g., links to published papers)?

All of the individual datasets are already released in OpenML, and many have been used in prior work on tabular data. However, our work gathers these datasets into a single 'hard' suite.

### What (other) tasks could the dataset be used for?

All of these datasets are tabular classification datasets, and so to the best of our knowledge, they cannot be used for anything other than tabular classification.

### Who funded the creation of the dataset? 

This benchmark suite was created by researchers at Abacus.AI, Stanford, Pinterest, University of Maryland, IIT Bombay, New York University, and Caltech. Funding for the dataset computation itself is from Abacus.AI.

### Any other comments?

None.

## Dataset Composition

### What are the instances?(that is, examples; e.g., documents, images, people, countries) Are there multiple types of instances? (e.g., movies, users, ratings; people, interactions between them; nodes, edges)

Each instance is a tabular datapoint. The makeup of each point depends on its dataset. For example, three of the datasets consist of poker hands, electricity usage, and plant textures.

### How many instances are there in total (of each type, if appropriate)?

See Table 3 in our paper for a breakdown of the number of instances for each dataset.

### What data does each instance consist of ? “Raw” data (e.g., unprocessed text or images)? Features/attributes? Is there a label/target associated with instances? If the instances related to people, are subpopulations identified (e.g., by age, gender, etc.) and what is their distribution?

The raw data is hosted on OpenML. In our repository, we also contain scripts for the standard preprocessing we ran before training tabular data models. The data are not related to people.

### Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.

There is no missing information.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.

There are no relationships between individual instances.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).

We selected the datasets for our benchmark suite as follows. We started with 176 datasets, which we selected with the aim to include most classification datasets from popular recent papers that study tabular data, including datasets from the OpenML-CC18 suite, the OpenML Benchmarking Suite, and additional OpenML datasets. Due to the scale of our experiments (538,650 total models trained), we limited to datasets smaller than 1.1M. CC-18 and OpenML Benchmarking Suite are both seen as the go-to standards for conducting a fair, diverse evaluation across algorithms due to their rigorous selection criteria and wide diversity of datasets.
Out of these 176 datasets, we selected 36 datasets for our suite as described in Section 3 of our paper.

### Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.

We use the 10 folds from OpenML, and it is recommended to report performance averaged over these 10 folds, as we do and as OpenML does. If a validation set is required, we recommend additionally using the validation splits that we used, described in Section 2 of our paper.

### Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.

There are no known errors, sources of noise, or redundancies.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

The dataset is self-contained.

### Any other comments?

None.


## Collection Process


### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?
 
We did not create the individual datasets. However, we selected the datasets for our benchmark suite as follows. We started with 176 datasets, which we selected with the aim to include most classification datasets from popular recent papers that study tabular data, including datasets from the OpenML-CC18 suite, the OpenML Benchmarking Suite, and additional OpenML datasets. Due to the scale of our experiments, we limited to datasets smaller than 1.1M. CC-18 and OpenML Benchmarking Suite are both seen as the go-to standards for conducting a fair, diverse evaluation across algorithms due to their rigorous selection criteria and wide diversity of datasets.

Out of these 176 datasets, we selected 36 datasets for our suite as described in Section 2 of our paper.

### How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.
 
The datasets were selected using the three criteria from Section 3 of our paper.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
 
As described earlier, the datasets were selected using the three criteria from Section 3 of our paper.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
 
The creation of the TabZilla Benchmark Suite was done by the authors of this work.

### Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.
 
The timeframe for constructing the TabZilla Benchmark Suite was from April 15, 2023 to June 1, 2023.

## Data Preprocessing

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.
 
We include both the raw data and the preprocessed data. We preprocessed the data by imputing each NaN to the mean of the respective feature. We left all other preprocessing (such as scaling) to the algorithms themselves, although we also ran experiments with additional preprocessing, which can be found in Appendix D of our paper.

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.

The raw data is available at https://www.openml.org/search?type=study&study_type=task&id=379&sort=tasks_included.

### Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.

Our README contains an extensive section on the data preprocessing, here: https://github.com/naszilla/tabzilla#openml-datasets.

### Does this dataset collection/processing procedure achieve the motivation for creating the dataset stated in the first section of this datasheet? If not, what are the limitations?
 
We hope that the release of this benchmark suite will achieve our goal of accelerating research in tabular data, as well as making it easier for researchers and practitioners to devise and compare algorithms. Time will tell whether our suite will be adopted by the community.

### Any other comments
 
None.

## Dataset Distribution

### How will the dataset be distributed? (e.g., tarball on website, API, GitHub; does the data have a DOI and is it archived redundantly?)
 
The benchmark suite is on OpenML at https://www.openml.org/search?type=study&study_type=task&id=379&sort=tasks_included.

### When will the dataset be released/first distributed? What license (if any) is it distributed under?
 
The benchmark suite is public as of June 1, 2023, distributed under the Apache License 2.0.

### Are there any copyrights on the data?
 
There are no copyrights on the data.

### Are there any fees or access/export restrictions?
 
There are no fees or restrictions.

### Any other comments?
 
None.

## Dataset Maintenance

### Who is supporting/hosting/maintaining the dataset?
 
The authors of this work are supporting/hosting/maintaining the dataset.

### Will the dataset be updated? If so, how often and by whom?
 
We welcome updates from the tabular data community. If new algorithms are created, the authors may open a pull request to include their method.

### How will updates be communicated? (e.g., mailing list, GitHub)
 
Updates will be communicated on the GitHub README: https://github.com/naszilla/tabzilla.

### If the dataset becomes obsolete how will this be communicated?
 
If the dataset becomes obsolete, it will be communicated on the GitHub README: https://github.com/naszilla/tabzilla.

### If others want to extend/augment/build on this dataset, is there a mechanism for them to do so? If so, is there a process for tracking/assessing the quality of those contributions. What is the process for communicating/distributing these contributions to users?
 
Others can create a pull request on GitHub with possible extensions to our benchmark suite, which will be approved case-by-case. For example, an author of a new hard tabular dataset may create a PR in our codebase with the new dataset. These updates will again be communicated on the GitHub README.


## Legal and Ethical Considerations

### Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.
 
There was no ethical review process. We note that our benchmark suite consists of existing datasets that are already publicly available on OpenML.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctorpatient confidentiality, data that includes the content of individuals non-public communications)? If so, please provide a description.
 
The datasets do not contain any confidential data.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.
 
None of the data might be offensive, insulting, threatening, or otherwise cause anxiety.

### Does the dataset relate to people? If not, you may skip the remaining questions in this section.
 
The datasets do not relate to people.

### Any other comments?
 
None.








