Aspect-Based Product Review Analysis
An NLP pipeline that turns raw Amazon phone reviews into feature-level sentiment and recurring-issue insights, explored through a Streamlit dashboard.

A star rating squeezes every opinion into one number. A 4-star phone can still have a great camera and a poor battery. This project scores each product feature (battery, camera, screen, price, network and more) separately and shows what customers complain about most.

Live app: https://appuct-review-analysis-qoq3pnzutqwcugkemobfix.streamlit.app/

What it does
Scores 43,230 aspect-sentence pairs across 14 aspects as positive, neutral or negative
Groups negative sentences into recurring issues (for example "Charging failure" or "SIM card missing or not working")
Aggregates results for 129 products, with mention counts so that noisy scores are visible
Offers a dashboard with product insights, side-by-side comparison, all-product trends and review search
Pipeline

Step	Technique	Model / tool
Overall sentiment	Pretrained transformer classifier	cardiffnlp/twitter-roberta-base-sentiment-latest
Sentence splitting	Rule/statistical sentence segmenter	spaCy en_core_web_sm
Aspect extraction	Keyword dictionary with word-boundary regex and exclusion phrases	Custom (14 aspects)
Aspect-level sentiment	Sentence-pair classification: the model reads (sentence, aspect) together	yangheng/deberta-v3-base-absa-v1.1
Issue mining	Sentence embeddings, per-aspect KMeans (5 clusters), TF-IDF keywords, hand-labeled clusters	all-MiniLM-L6-v2, scikit-learn
Review search	Cosine similarity over sentence embeddings, with TF-IDF fallback	all-MiniLM-L6-v2, scikit-learn
Dashboard	Interactive app over precomputed CSV files	Streamlit, Altair
All pretrained models are used as released. No model was fine-tuned in this project.

Results
Evaluated on a blind, hand-labeled random sample of 200 aspect-sentence pairs. I labeled the rows with the model's predictions hidden. Each row got two labels: whether the sentence is really about the aspect, and its sentiment.

Measure	Result
Aspect detection precision (n = 200)	78.5% (approx. 95% interval 72.8% to 84.2%)
Sentiment accuracy (157 relevant sentences)	82.8%
Sentiment macro-F1 (157 relevant sentences)	0.667
Threshold rule for neutral predictions, macro-F1	0.645, so it was not adopted
Overall sentiment, checked against star ratings: 83% of 1-star reviews were labeled negative and 88% of 5-star reviews positive.

Why SemEval/MAMS are not used as benchmarks: the DeBERTa ABSA model was already trained on both, so scores there would not be independent evidence of quality. The hand-labeled Amazon sample is the evaluation.

A negative result worth reporting: I tested a probability-threshold rule meant to fix the model's tendency to predict neutral for implied opinions. The rule was decided in advance (adopt only if its blind-test macro-F1 is not below the baseline's). It scored lower, so the plain argmax prediction is used.

Limitations
Aspect precision is 78.5%. Battery and customer service were the weakest aspects in the sample.
Recall of the keyword matcher was not measured. Dropping generic words such as "app" and "android" improved software precision but removed some real complaints.
The blind test has only 9 neutral examples, so neutral-class performance is not reliably measured.
A single annotator labeled all data, and issue-theme names were assigned by hand.
The model tends to predict neutral for implied opinions (for example, a complaint stated indirectly).
Semantic search matches topic more than tone or negation. Use the Sentiment filter to separate complaints from praise.
Results come from one dataset of mostly older phones and may not carry over to other product categories.
Repository structure
.
├── README.md
├── dashboard/
│   ├── app.py                  Streamlit app
│   ├── requirements.txt
│   └── data/                   Precomputed outputs used by the app
├── notebooks/
│   ├── 02-aspects.ipynb        Aspect extraction and aspect-level sentiment
│   ├── 03-themes.ipynb         Issue mining (first run)
│   ├── 04-eval.ipynb           Evaluation, threshold experiments, blind test
│   ├── 05updatedthemes.ipynb   Issue labels carried over after the matcher fix
│   └── 06-search-index.ipynb   Sentence index and embeddings for search
└── .gitignore
Run the dashboard locally
git clone https://github.com/hiteshi0338/product-review-analysis.git
cd product-review-analysis
python -m venv venv
venv\Scripts\activate            # Windows (use: source venv/bin/activate on Mac/Linux)
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
The dashboard reads precomputed CSV files from dashboard/data/, so no GPU is needed. The Search tab uses semantic search when sentence-transformers is installed and falls back to TF-IDF keyword search otherwise.

Reproduce the pipeline
The heavy steps (sentiment models and embeddings) were run in Kaggle GPU notebooks. Run the notebooks in this order:

Data loading and overall sentiment
notebooks/02-aspects.ipynb
notebooks/04-eval.ipynb
notebooks/03-themes.ipynb, then notebooks/05updatedthemes.ipynb
Aggregation
notebooks/06-search-index.ipynb
Update the /kaggle/input/... paths at the top of each notebook to match your own Kaggle inputs.

Tech stack
Python, pandas, NumPy, scikit-learn, spaCy, Hugging Face Transformers, sentence-transformers, Altair, Streamlit. GPU inference on Kaggle.

Data
Amazon cell-phone reviews: https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews . The raw dataset is not included in this repository. Review sentences in dashboard/data/ are used for demonstration; check the dataset's terms before reusing them.
