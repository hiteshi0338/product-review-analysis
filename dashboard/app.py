import pandas as pd
import altair as alt
import streamlit as st
import numpy as np

st.set_page_config(page_title="Product Review Intelligence", layout="wide")


@st.cache_data
def load():
    scores = pd.read_csv("data/product_aspect_scores.csv")
    overall = pd.read_csv("data/overall_aspect_summary.csv")
    issues = pd.read_csv("data/product_issues.csv")
    ex_issue = pd.read_csv("data/issue_examples.csv")
    ex = pd.read_csv("data/examples_by_product.csv")
    return scores, overall, issues, ex_issue, ex

@st.cache_resource
def load_search():
    idx = pd.read_csv("data/search_index.csv")
    emb = np.load("data/search_emb.npy").astype("float32")
    S = {"idx": idx, "emb": emb, "mode": "keyword", "model": None}
    try:
        from sentence_transformers import SentenceTransformer
        S["model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        S["mode"] = "semantic"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        S["tfidf"] = tf
        S["X"] = tf.fit_transform(idx["sentence"])
    return S

def run_search(S, query, asin, aspect, sentiment, k, min_sim, allowed_asins):
    idx = S["idx"]
    mask = idx["asin"].isin(allowed_asins).to_numpy(copy=True)

    if asin:
        mask = mask & (idx["asin"] == asin).to_numpy()
    if aspect != "Any":
        mask = mask & idx["aspect_list"].str.contains(
            rf"(?:^|\|){aspect}(?:\||$)", regex=True).to_numpy()
    if sentiment != "Any":
        mask = mask & idx["sentiments"].str.contains(sentiment, regex=False).to_numpy()

    if S["mode"] == "semantic":
        qv = S["model"].encode([query], normalize_embeddings=True)[0]
        sims = S["emb"] @ qv
    else:
        qv = S["tfidf"].transform([query])
        sims = (S["X"] @ qv.T).toarray().ravel()

    sims = np.where(mask, sims, -1.0)

    order = np.argsort(-sims)[: k * 4]
    cand = [i for i in order if sims[i] >= min_sim]
    out = idx.iloc[cand].copy()
    out["similarity"] = sims[cand]
    out["_norm"] = out["sentence"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    out = out.drop_duplicates("_norm").head(k).drop(columns="_norm")
    return out

scores, overall, issues, ex_issue, ex = load()
TH = "theme" if "theme" in issues.columns else "theme_label"
TH_EX = "theme" if "theme" in ex_issue.columns else "theme_label"

scores["product"] = scores["product"].fillna(scores["asin"])
scores["brand"] = scores["brand"].fillna("Unknown")

products = (scores.groupby("asin")
            .agg(product=("product", "first"), brand=("brand", "first"), mentions=("mentions", "sum"))
            .reset_index().sort_values("mentions", ascending=False))
products["label"] = products["product"].str.slice(0, 70) + "  (" + products["mentions"].astype(str) + " mentions)"


def score_chart(df, title=""):
    d = df.sort_values("net_score", ascending=False)
    return (alt.Chart(d).mark_bar().encode(
        x=alt.X("net_score:Q", scale=alt.Scale(domain=[-1, 1]),
                title="Net sentiment (positive share minus negative share)"),
        y=alt.Y("aspect:N", sort=list(d["aspect"]), title=None),
        color=alt.condition(alt.datum.net_score > 0, alt.value("#2a9d8f"), alt.value("#e76f51")),
        tooltip=["aspect", "mentions",
                 alt.Tooltip("pos:Q", format=".0%"), alt.Tooltip("neg:Q", format=".0%"), "net_score"],
    ).properties(title=title, height=30 * len(d) + 40))


st.title("Product Review Intelligence")
st.caption("Aspect-level sentiment and recurring issues mined from Amazon phone reviews.")

with st.sidebar:
    st.header("Select a product")
    brands = ["All"] + sorted(products["brand"].unique())
    brand = st.selectbox("Brand", brands)
    pool = products if brand == "All" else products[products["brand"] == brand]
    choice = st.selectbox("Product", pool["label"])
    asin = pool.loc[pool["label"] == choice, "asin"].iloc[0]
    st.caption("Aspect scores with few mentions are noisy; check the 'mentions' column.")

tab1, tab2, tab3, tab4 = st.tabs(["Product insights", "Compare products", "All products", "Search reviews"])


with tab1:
    d = scores[scores["asin"] == asin]
    name = d["product"].iloc[0]
    st.subheader(name)

    c1, c2, c3 = st.columns(3)
    c1.metric("Aspect mentions", int(d["mentions"].sum()))
    c2.metric("Best aspect", d.sort_values("net_score").iloc[-1]["aspect"])
    c3.metric("Weakest aspect", d.sort_values("net_score").iloc[0]["aspect"])

    st.altair_chart(score_chart(d), use_container_width=True)
    st.dataframe(d[["aspect", "mentions", "pos", "neg", "net_score"]]
                 .sort_values("net_score", ascending=False)
                 .style.format({"pos": "{:.0%}", "neg": "{:.0%}", "net_score": "{:+.2f}"}),
                 hide_index=True, use_container_width=True)

    st.markdown("#### Recurring issues for this product")
    iss = (issues[issues["asin"] == asin].groupby(["aspect", TH])["mentions"].sum()
           .reset_index().sort_values("mentions", ascending=False).head(8))
    if iss.empty:
        st.info("Not enough negative mentions to show product-specific issues. See the 'All products' tab.")
    else:
        st.dataframe(iss.rename(columns={TH: "issue"}), hide_index=True, use_container_width=True)

    st.markdown("#### Example reviews")
    asp = st.selectbox("Aspect", d.sort_values("mentions", ascending=False)["aspect"])
    e = ex[(ex["asin"] == asin) & (ex["aspect"] == asp)]
    left, right = st.columns(2)
    left.markdown("**Positive**")
    for s in e[e["sentiment"] == "positive"]["sentence"]:
        left.write(f"- {s}")
    right.markdown("**Negative**")
    for s in e[e["sentiment"] == "negative"]["sentence"]:
        right.write(f"- {s}")

with tab2:
    st.write("Compare two products aspect by aspect.")
    ca, cb = st.columns(2)
    la = ca.selectbox("Product A", products["label"], index=0, key="a")
    lb = cb.selectbox("Product B", products["label"], index=min(1, len(products) - 1), key="b")
    ida = products.loc[products["label"] == la, "asin"].iloc[0]
    idb = products.loc[products["label"] == lb, "asin"].iloc[0]

    a = scores[scores["asin"] == ida].assign(who="A: " + la.split("  (")[0][:40])
    b = scores[scores["asin"] == idb].assign(who="B: " + lb.split("  (")[0][:40])
    both = pd.concat([a, b])
    common = set(a["aspect"]) & set(b["aspect"])
    both = both[both["aspect"].isin(common)]

    if both.empty:
        st.info("These two products share no aspects with enough mentions.")
    else:
        chart = (alt.Chart(both).mark_bar().encode(
            x=alt.X("net_score:Q", scale=alt.Scale(domain=[-1, 1]), title="Net sentiment"),
            y=alt.Y("aspect:N", title=None),
            yOffset="who:N", color="who:N",
            tooltip=["who", "aspect", "mentions", "net_score"],
        ).properties(height=40 * len(common) + 60))
        st.altair_chart(chart, use_container_width=True)

with tab3:
    st.subheader("All products")
    st.altair_chart(score_chart(overall), use_container_width=True)

    st.markdown("#### Most common issues")
    asp2 = st.selectbox("Aspect", sorted(issues["aspect"].unique()), key="asp_all")
    top = (issues[issues["aspect"] == asp2].groupby(TH)["mentions"].sum()
           .reset_index().sort_values("mentions", ascending=False))
    st.dataframe(top.rename(columns={TH: "issue"}), hide_index=True, use_container_width=True)

    th = st.selectbox("See examples for issue", top[TH], key="theme_all")
    for s in ex_issue[(ex_issue["aspect"] == asp2) & (ex_issue[TH_EX] == th)]["sentence"].head(5):
        st.write(f"- {s}")

with tab4:
    st.subheader("Search reviews by meaning")
    S = load_search()
    if S["mode"] == "semantic":
        st.caption("Semantic search: finds sentences with a similar meaning, even if they use different words.")
    else:
        st.caption("Keyword search mode (the semantic model could not be loaded on this machine).")

    q = st.text_input("Ask something about the phones",
                      placeholder="e.g. does it overheat?  good for travel?  screen cracks easily")
    st.caption("Tip: matching is by topic, not tone. Use the Sentiment filter to separate complaints from praise.")

    c1, c2, c3, c4 = st.columns(4)
    scope = c1.radio("Search in", ["Selected product", "All products"])
    aspect_opt = c2.selectbox("Aspect", ["Any"] + sorted(scores["aspect"].unique()))
    sent_opt = c3.selectbox("Sentiment", ["Any", "negative", "positive", "neutral"])
    k = c4.slider("Results", 5, 30, 10)
    min_sim = st.slider("Minimum similarity", 0.0, 0.8,
                        0.30 if S["mode"] == "semantic" else 0.05, 0.01,
                        help="Raise it for stricter matches, lower it to see more results.")

    if q.strip():
        asin_f = asin if scope == "Selected product" else None
        res = run_search(S, q.strip(), asin_f, aspect_opt, sent_opt, k, min_sim, set(products["asin"]))

        if res.empty:
            st.info("No matches. Try fewer filters, other wording, or a lower minimum similarity.")
        else:
            neg_share = res["sentiments"].str.contains("negative").mean()
            pos_share = res["sentiments"].str.contains("positive").mean()
            m1, m2, m3 = st.columns(3)
            m1.metric("Matches", len(res))
            m2.metric("Tagged negative", f"{neg_share:.0%}")
            m3.metric("Tagged positive", f"{pos_share:.0%}")

            name_map = products.set_index("asin")["product"].to_dict()
            for _, r in res.iterrows():
                title = name_map.get(r["asin"], r["asin"])
                st.markdown(f"> {r['sentence']}")
                st.caption(f"{title[:70]}  |  {r['aspects']}  |  {int(r['rating'])} stars  |  similarity {r['similarity']:.2f}")
    else:
        st.write("Try: *battery dies quickly*, *good camera in low light*, *arrived damaged*, *worth the price*.")

with st.expander("About this analysis"):
    st.markdown("""
- Pipeline: review cleaning, keyword-based aspect extraction, DeBERTa aspect sentiment, embedding + clustering of negative sentences into issues.
- Evaluated on a blind, hand-labeled random sample: aspect precision about 78.5% (n = 200); sentiment accuracy about 83% and macro-F1 0.67 on relevant sentences (n = 157).
- Limitations: neutral had only 9 test examples, so neutral performance is not established; keyword matching misses some software complaints; scores based on few mentions are noisy.
""")