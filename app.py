import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


@st.cache_data
def load_data(path):
    df = pd.read_excel(path, header=3)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in ["Дата регистрации", "Дата рождения"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    lower = df["Сумма договора"].quantile(0.01)
    upper = df["Сумма договора"].quantile(0.99)
    df["Сумма договора"] = df["Сумма договора"].clip(lower, upper)
    ir = get_interest_rate(df, "Сумма договора", "Дата регистрации")
    df["Сумма договора (disc)"] = discounting(
        df, ir, "Сумма договора", "Дата регистрации"
    )
    df["Email"] = df["Email"].str.split("@").str[-1]
    df["Пол"] = df["Контрагент"].apply(extract_gender)
    cols = [
        "Вид помещения",
        "Контрагент",
        "Дата рождения",
        "Телефон",
        "Email",
        "Объект строительства",
        "Проект",
        "Текущий регион",
        "Дата регистрации",
        "Типология",
        "Сумма договора",
        "Площадь договора",
        "Сумма договора (disc)",
        "Пол",
    ]
    return df[cols]


def get_interest_rate(
    deals, target_name="Сумма договора", date_col="Дата регистрации", is_ml=True
):
    df = deals.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col)
    monthly = df.set_index(date_col)[target_name].resample("M").mean().to_period("M")
    pct = monthly.pct_change()
    if is_ml:
        pct = pct.shift()
    pct = pct.fillna(0)
    return (1 + pct).cumprod()


def discounting(
    deals, rates, target_name="Сумма договора", date_col="Дата регистрации"
):
    df = deals.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    periods = df[date_col].dt.to_period("M")
    factors = periods.map(lambda p: rates.get(p, 1.0))
    return df[target_name] / factors


def guess_gender_by_patronymic(p):
    if not isinstance(p, str) or not p.strip():
        return "Не определено"
    s = p.lower()
    if s.endswith(("ич", "вич")):
        return "Мужской"
    if s.endswith(("вна", "чна", "шна", "ина", "инична", "евна", "ёвна")):
        return "Женский"
    return "Не определено"


def extract_gender(name):
    if not isinstance(name, str):
        return "Не определено"
    parts = name.split()
    return guess_gender_by_patronymic(parts[2]) if len(parts) == 3 else "Не определено"


def main():
    df = load_data("Список.xlsx")
    dependent = [
        c
        for c in ["Сумма договора (disc)", "Сумма договора", "Площадь договора"]
        if c in df.columns
    ]
    features = [c for c in df.columns if c not in dependent]
    dep = st.sidebar.selectbox("Целевая", dependent)
    feat = st.sidebar.selectbox("Признак", features)
    regions = sorted(df["Текущий регион"].dropna().unique())
    types = sorted(df["Вид помещения"].dropna().unique())
    cities = st.sidebar.multiselect("Регион", ["Все"] + regions, default="Все")
    room_types = st.sidebar.multiselect("Вид помещения", ["Все"] + types, default="Все")
    bins = st.sidebar.slider("Bins", 5, 100, 30, 5)
    trim = st.sidebar.checkbox("Обрезать 1–99%", False)
    normalize = st.sidebar.checkbox("Нормализовать", False)
    data = df.copy()
    if "Все" not in cities:
        data = data[data["Текущий регион"].isin(cities)]
    if "Все" not in room_types:
        data = data[data["Вид помещения"].isin(room_types)]
    norm = "percent" if normalize else None
    if is_datetime64_any_dtype(data[feat]):
        data["year"] = data[feat].dt.year
        avg = data.groupby("year")[dep].mean().reset_index()
        st.plotly_chart(px.line(avg, x="year", y=dep))
        st.plotly_chart(px.histogram(data, x="year", nbins=bins, histnorm=norm))
        data["decile"] = pd.qcut(data[feat], 10).astype(str)
        fig = px.histogram(
            data, x=dep, facet_col="decile", facet_col_wrap=5, nbins=bins, histnorm=norm
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig)
    elif is_numeric_dtype(data[feat]):
        x = data[feat]
        if trim:
            ql, qh = x.quantile([0.01, 0.99])
            x = x[(x >= ql) & (x <= qh)]
        st.plotly_chart(px.scatter(data.assign(x=x), x="x", y=dep, trendline="lowess"))
        st.plotly_chart(px.histogram(x=x, nbins=bins, histnorm=norm))
        st.plotly_chart(px.box(y=x))
    else:
        s = data[feat].astype(str)
        cnt = s.value_counts()
        if len(cnt) > 10:
            top = cnt.nlargest(10).index
            data["grp"] = s.where(s.isin(top), other="Прочие")
            cats = list(top) + ["Прочие"]
        else:
            data["grp"] = s
            cats = cnt.index.tolist()
        bar = data.groupby("grp")[dep].mean().reindex(cats).reset_index()
        st.plotly_chart(px.bar(bar, x="grp", y=dep, category_orders={"grp": cats}))
        cntdf = data["grp"].value_counts().reindex(cats).reset_index()
        cntdf.columns = ["grp", "count"]
        st.plotly_chart(
            px.bar(cntdf, x="grp", y="count", category_orders={"grp": cats})
        )
        fig = px.histogram(
            data.assign(grp=data["grp"]),
            x=dep,
            facet_col="grp",
            facet_col_wrap=4,
            nbins=bins,
            histnorm=norm,
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig)
        st.plotly_chart(
            px.box(
                data.assign(grp=data["grp"]),
                x="grp",
                y=dep,
                category_orders={"grp": cats},
            )
        )


if __name__ == "__main__":
    main()
