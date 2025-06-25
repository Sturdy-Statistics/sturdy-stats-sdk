# Sturdy Stats SDK

This is the sdk for the [Sturdy Statistics API](https://sturdystatistics.com/). We host a [series of public indicies](https://sturdystatistics.com/gallery) trained on Earnings Calls, ArXiv, HackerNews, and various news streams that anyone can use for public data analysis. Uploading data requires signing up at https://sturdystatistics.com in order to create an api key. 

## Installation

Core API: `pip install sturdy-stats-sdk`

Regression Extension: `pip install sturdy-stats-sdk[regression]`

## Resources

Explore our [gallery](https://sturdystatistics.com/gallery/) to browse visualization created by the sturdy-stats-sdk. Follow along with our [quickstart](https://sturdystatistics.com/docs) to hit the ground running or browse our [advanced examples](https://sturdystatistics.com/docs/examples/) to perform rigorous analyses.

## Technical Features

<dl><dt>Automatic Structuring of Unstructured Text Data</dt><span></span><dd>Convert unstructured documents into structured formats, allowing seamless analysis alongside traditional tabular data.<a href="https://sturdystatistics.com/features.html#structure"> Learn More &gt;</a></dd><span></span><dt>Explainable Text Classification</dt><span></span><dd>Gain clear insights into how text data is categorized, while enhancing transparency and trust in your analyses.<a href="https://sturdystatistics.com/features.html#classification"> Learn More &gt;</a></dd><span></span><dt>Effective with Small Datasets</dt><span></span><dd>Achieve meaningful results even with limited data, making our solutions accessible to organizations of all sizes.<a href="https://sturdystatistics.com/features.html#sparse-prior"> Learn More &gt;</a></dd><span></span><dt>Powerful Search Capabilities</dt><span></span><dd>Leverage our robust search API to retrieve and analyze specific information within your unstructured data.<a href="https://sturdystatistics.com/features.html#search"> Learn More &gt;</a></dd><span></span><dt>Comprehensive Data Lake</dt><span></span><dd>Store and analyze all your data — structured and unstructured — in one place, facilitating holistic insights.<a href="https://sturdystatistics.com/features.html#data-lake"> Learn More &gt;</a></dd><span></span></dl>

## Quickstart

#### Explore Your Data
```python
from sturdystats import Index, Job
import plotly.express as px

index = Index(id="index_99051ff1489844878fd792784d7baa90")
topic_df = index.topicSearch()
fig = px.sunburst(
    topic_df, 
    path=["topic_group_short_title", "short_title"],
    values="prevalence", 
    hover_data=["topic_id"]
)
```


#### Run SQL queries against your unstructured ata
```python
topic_id = 12
df = pd.DataFrame(index.queryMeta(f"""
SELECT
    quarter,
    sum(sparse_list_extract({topic_id+1}, sum_topic_counts_inds, sum_topic_counts_vals)) as n_occurences
FROM doc 
GROUP BY quarter 
ORDER BY quarter""") )
```

#### Create a Index from scratch
```python
from sturdystats import Index, Job
import pandas as pd

df = pd.read_parquet('data.parquet')
index = Index(API_key="XXX", name='tech_earnings_calls_2024')

res = index.upload(df.to_dict("records"))
job = index.train(params=dict(), fast=True, wait=True)
```

#### Train robust linear models.

`pip install sturdy-stats-sdk[regression]`

```python
from sturdystats.model import LinearRegressor 
import arviz as az

model = LinearRegression(API_key=API_KEY)
model.sample(X, Y) 
az.plot_trace(model.inference_data)
```

#### Detect mislabelled datapoints.
```python
from sturdystats.model import SturdyLogisticRegressor
import arviz as az

model = SturdyLogisticRegressor(API_key=API_KEY)
model.sample(X, Y) 
az.plot_trace(model.inference_data)
```
