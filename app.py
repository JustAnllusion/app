import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in ("Дата регистрации", "Дата рождения"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    if {"Дата регистрации", "Дата рождения"}.issubset(df.columns):
        df["Возраст"] = (
            ((df["Дата регистрации"] - df["Дата рождения"]).dt.days / 365.25)
            .floordiv(1)
            .astype("Int64")
        )
    if "Сумма договора" in df.columns:
        low, high = df["Сумма договора"].quantile([0.01, 0.99])
        df["Сумма договора"] = df["Сумма договора"].clip(low, high)
    if "Email" in df.columns:
        df["Email"] = df["Email"].str.split("@").str[-1]
    if "Контрагент" in df.columns:
        df["Пол"] = df["Контрагент"].apply(extract_gender)
    return df

def discounting(deals, rates, target_name, date_col):
    periods = deals[date_col].dt.to_period("M")
    factors = periods.map(lambda p: rates.get(p, 1.0))
    return deals[target_name] / factors

def guess_gender_by_patronymic(p):
    if not isinstance(p, str) or not p.strip():
        return "Не определено"
    s = p.lower()
    if s.endswith(("ич", "вич")):
        return "Мужской"
    if s.endswith(("вна", "чина", "шна", "ина", "инична", "евна", "ёвна")):
        return "Женский"
    return "Не определено"

def extract_gender(name):
    parts = str(name).split()
    return guess_gender_by_patronymic(parts[2]) if len(parts) == 3 else "Не определено"

def main():
    df = load_data("df_with_polygon.xlsx")
    df["Текущий регион"] = df["Текущий регион"].astype(str).str.title()
    if "Типология" in df.columns:
        df = df[df["Типология"] != "сп"]

    section = st.sidebar.radio("Раздел", ["Обзор признаков", "Анализ данных"], key="sec")
    regions = sorted(df["Текущий регион"].dropna().unique())
    selected_regions = st.sidebar.multiselect("Регион", ["Все"] + regions, default=["Все"], key="region")
    if "Все" not in selected_regions:
        df = df[df["Текущий регион"].isin(selected_regions)]
    selected_clusters = []

    cluster_map = {
        1: "1-ый кластер",
        2: "2-ой кластер",
        3: "3-ий кластер",
       -1: "4: выбросы по квадратным метрам",
        4: "5: выброс по цене за метр",
        5: "6: выброс по общей цене",
        6: "7: выбросы и по цене за метр и по общей цене",
    }
    if section == "Анализ данных" and "polygon_id" in df.columns and selected_regions == ["Новосибирск"]:
        df = df[df["polygon_id"] != -2].copy()
        df["Кластер"] = df["polygon_id"].map(cluster_map).fillna("Не определено")
        all_clusters = sorted(df["Кластер"].unique())
        default_clusters = ["1-ый кластер", "2-ой кластер", "3-ий кластер"]
        selected_clusters = st.sidebar.multiselect("Кластер", all_clusters, default=default_clusters, key="cluster")
        df = df[df["Кластер"].isin(selected_clusters)]
        if "Вид помещения" in df.columns:
            df = df[df["Вид помещения"].str.lower() == "квартира"]

    if section == "Анализ данных" and selected_regions == ["Новосибирск"]:
        colorize = st.sidebar.checkbox("Раскрасить по кластерам", key="colorize")
    else:
        colorize = False

    if {"Сумма договора", "Дата регистрации"}.issubset(df.columns):
        price_index = (
            df.set_index("Дата регистрации")["Сумма договора"]
            .resample("M")
            .mean()
            .to_period("M")
            .pct_change()
            .fillna(0)
            .add(1)
            .cumprod()
        )
        first_factor = price_index / price_index.iloc[0]
        last_factor  = price_index / price_index.iloc[-1]

        def _to_base(row, factors):
            dt = row["Дата регистрации"]
            if pd.isna(dt):
                return np.nan
            period = dt.to_period("M")
            return row["Сумма договора"] / factors.get(period, 1.0)

        df["Сумма договора (к первой дате)"] = df.apply(_to_base, axis=1, factors=first_factor)
        df["Сумма договора (к последней дате)"] = df.apply(_to_base, axis=1, factors=last_factor)

    targets = [
        c
        for c in (
            "Сумма договора",
            "Сумма договора (к первой дате)",
            "Сумма договора (к последней дате)",
            "Площадь договора",
        )
        if c in df.columns
    ]

    all_feats = [
        c
        for c in (
            "Кластер",
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
            "Возраст",
        )
        if c in df.columns
    ]
    analysis_feats = [
        c
        for c in all_feats
        if c not in ("Телефон", "Вид помещения", "Контрагент", "Текущий регион")
    ]

    if section == "Обзор признаков":
        st.header("Обзор признаков и целевых метрик")
        bins = st.sidebar.slider("Количество бинов", 5, 100, 30, 5)
        normalize = st.sidebar.checkbox("Нормализовать распределения")
        remove_outliers = st.sidebar.checkbox("Удалить выбросы")
        histnorm = "percent" if normalize else None

        for col in all_feats + targets:
            st.subheader(col)
            data_col = df[col]

            if remove_outliers:
                if is_numeric_dtype(data_col):
                    low, high = data_col.quantile([0.01, 0.99])
                    data_col = data_col[data_col.between(low, high)]
                elif is_datetime64_any_dtype(data_col):
                    dt = data_col.dropna()
                    dt_int = dt.astype("int64")
                    low_i, high_i = np.quantile(dt_int, [0.01, 0.99])
                    data_col = data_col[data_col.between(pd.to_datetime(low_i), pd.to_datetime(high_i))]

            c1, c2 = st.columns(2)

            if col == "Телефон":
                vals = (
                    data_col.astype(str)
                    .str.replace(r"\D", "", regex=True)
                    .str[:3]
                )
                counts = vals.value_counts()
                top = counts.index if len(counts) <= 10 else counts.nlargest(10).index
                grp = vals.where(vals.isin(top), "Прочие")
                cnt = grp.value_counts().reindex(
                    list(top) + (["Прочие"] if len(counts) > 10 else [])
                )
                with c1:
                    st.write(cnt)
                with c2:
                    fig = px.bar(x=cnt.index, y=cnt.values)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

            elif is_numeric_dtype(data_col):
                with c1:
                    st.write(data_col.describe())
                with c2:
                    x = data_col.dropna()
                    fig = px.histogram(x=x, nbins=bins, histnorm=histnorm)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

            elif is_datetime64_any_dtype(data_col):
                with c1:
                    try:
                        stats = data_col.describe(datetime_is_numeric=True)
                    except TypeError:
                        stats = data_col.describe()
                    st.write(stats)
                with c2:
                    dates = data_col.dropna()
                    months = dates.dt.to_period("M").dt.to_timestamp()
                    fig = px.histogram(x=months, nbins=bins, histnorm=histnorm)
                    fig.update_layout(
                        height=200, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Месяц"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                vals = data_col.astype(str).value_counts()
                top_vals = vals.nlargest(10).index.tolist() if len(vals) > 10 else vals.index.tolist()
                grp = data_col.astype(str).where(data_col.astype(str).isin(top_vals), "Прочие")
                cats = sorted(top_vals) + (["Прочие"] if len(vals) > 10 else [])
                cnt = grp.value_counts().reindex(cats)
                with c1:
                    st.write(cnt)
                with c2:
                    fig = px.bar(x=cnt.index, y=cnt.values)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.header("Анализ данных")
        target = st.sidebar.selectbox("Целевая переменная", targets)
        feature = st.sidebar.selectbox("Признак", analysis_feats)
        bins = st.sidebar.slider("Количество бинов", 5, 100, 30, 5)
        normalize = st.sidebar.checkbox("Нормализовать распределения")
        histnorm = "percent" if normalize else None

        data = df.copy()
        remove_feature = st.sidebar.checkbox("Удалить выбросы по признаку")
        remove_target = st.sidebar.checkbox("Удалить выбросы по таргету")
        if remove_feature:
            if is_datetime64_any_dtype(data[feature]):
                dt_int = data[feature].dropna().astype("int64")
                low_i, high_i = np.quantile(dt_int, [0.01, 0.99])
                data = data[data[feature].between(pd.to_datetime(low_i), pd.to_datetime(high_i))]
            elif is_numeric_dtype(data[feature]):
                vals = data[feature].dropna()
                low_f, high_f = vals.quantile([0.01, 0.99])
                data = data[data[feature].between(low_f, high_f)]
        if remove_target and is_numeric_dtype(data[target]):
            tvals = data[target].dropna()
            low_t, high_t = tvals.quantile([0.01, 0.99])
            data = data[data[target].between(low_t, high_t)]

        col1, col2 = st.columns(2)

        if is_datetime64_any_dtype(data[feature]) or feature == "Возраст":
            if is_datetime64_any_dtype(data[feature]):
                x_val = data[feature].dropna().dt.to_period("M").dt.to_timestamp()
            else:
                x_val = data[feature].dropna()
            df_raw = data.assign(x=x_val).dropna(subset=["x", target])
            is_date_feature = is_datetime64_any_dtype(data[feature])
            if is_date_feature:
                df_raw["year"] = df_raw["x"].dt.to_period("Y").dt.to_timestamp()

            st.subheader("Scatter")
            color_arg = "Кластер" if colorize and "Кластер" in data.columns else None
            st.plotly_chart(
                px.scatter(df_raw, x="x", y=target, color=color_arg)
                .update_layout(xaxis_title=feature, yaxis_title=target),
                use_container_width=True,
            )

            st.subheader("Копула")
            x_q = x_val.rank(method="average", pct=True)
            t_q = data[target].rank(method="average", pct=True)
            if colorize and "Кластер" in data.columns:
                cop = pd.DataFrame({"x_q": x_q, "t_q": t_q, "Кластер": data["Кластер"]}).dropna()
                fig = px.scatter(
                    cop,
                    x="x_q",
                    y="t_q",
                    color="Кластер",
                    labels={"x_q": f"Квантиль {feature} (pct)", "t_q": f"Квантиль {target} (pct)"}
                )
            else:
                cop = pd.DataFrame({"x_q": x_q, "t_q": t_q}).dropna()
                fig = px.scatter(
                    cop,
                    x="x_q",
                    y="t_q",
                    labels={"x_q": f"Квантиль {feature} (pct)", "t_q": f"Квантиль {target} (pct)"}
                )
            fig.update_layout(
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            if is_date_feature:
                with col1:
                    st.subheader("Среднее по неделям")
                    df_raw["week"] = df_raw["x"].dt.to_period("W").dt.start_time
                    if colorize and "Кластер" in data.columns:
                        mean_week = df_raw.groupby(["week", "Кластер"], as_index=False)[target].mean()
                        fig_week = px.scatter(mean_week, x="week", y=target, color="Кластер")
                    else:
                        mean_week = df_raw.groupby("week", as_index=False)[target].mean()
                        fig_week = px.scatter(mean_week, x="week", y=target)
                    fig_week.update_layout(xaxis_title="Неделя", yaxis_title=target)
                    fig_week.update_yaxes(tickformat=".3~s")
                    st.plotly_chart(fig_week, use_container_width=True)
            else:
                with col1:
                    st.subheader("Среднее по десятилетиям возраста")
                    age_dec = df_raw.assign(age_dec=(df_raw["x"] // 10) * 10)
                    mean_dec = age_dec.groupby("age_dec", as_index=False)[target].mean()
                    fig_dec = px.scatter(mean_dec, x="age_dec", y=target)
                    fig_dec.update_layout(xaxis_title="Возраст (округл. до 10 лет)", yaxis_title=target)
                    fig_dec.update_yaxes(tickformat=".3~s")
                    st.plotly_chart(fig_dec, use_container_width=True)

            with col2:
                if is_date_feature:
                    st.subheader("Среднее по году")
                    if colorize and "Кластер" in data.columns:
                        mean_year = df_raw.groupby(["year", "Кластер"], as_index=False)[target].mean()
                        fig_year = px.line(mean_year, x="year", y=target, color="Кластер")
                    else:
                        mean_year = df_raw.groupby("year", as_index=False)[target].mean()
                        fig_year = px.line(mean_year, x="year", y=target)
                    fig_year.update_layout(xaxis_title="Год", yaxis_title=target)
                    fig_year.update_yaxes(tickformat=".3~s")
                    st.plotly_chart(fig_year, use_container_width=True)
                else:
                    st.subheader("Среднее по возрасту (один год)")
                    mean_age = df_raw.groupby("x", as_index=False)[target].mean()
                    fig_age = px.line(mean_age, x="x", y=target)
                    fig_age.update_layout(xaxis_title="Возраст", yaxis_title=target)
                    fig_age.update_yaxes(tickformat=".3~s")
                    st.plotly_chart(fig_age, use_container_width=True)

            st.subheader(f"Распределение {feature}")
            if colorize:
                hist_df = data.assign(x=x_val).dropna(subset=["x"])
                fig = px.histogram(hist_df, x="x", color="Кластер", nbins=bins, histnorm=histnorm)
            else:
                fig = px.histogram(x=x_val, nbins=bins, histnorm=histnorm)
            fig.update_layout(xaxis_title=feature, yaxis_title="count")
            st.plotly_chart(fig, use_container_width=True)

            if feature != "Возраст":
                x_num = x_val.view("int64")
            else:
                x_num = x_val
            edges = pd.qcut(x_num, 10, retbins=True, duplicates="drop")[1]
            if feature != "Возраст":
                edges = pd.to_datetime(edges)
            labels = []
            for i in range(len(edges) - 1):
                l, r = edges[i], edges[i + 1]
                labels.append(
                    f"{int(l)}-{int(r)}" if feature == "Возраст" else f"{l.strftime('%Y-%m')}-{r.strftime('%Y-%m')}"
                )
            data_dec = data.assign(x=x_val)
            data_dec["decile"] = pd.cut(x_val, bins=edges, labels=labels, include_lowest=True).dropna()

            st.subheader("Таргет по интервалам")
            if colorize:
                fig = px.histogram(
                    data_dec,
                    x=target,
                    color="Кластер",
                    facet_col="decile",
                    facet_col_wrap=5,
                    nbins=bins,
                    histnorm=histnorm,
                    category_orders={"decile": labels},
                )
            else:
                fig = px.histogram(
                    data_dec,
                    x=target,
                    facet_col="decile",
                    facet_col_wrap=5,
                    nbins=bins,
                    histnorm=histnorm,
                    category_orders={"decile": labels},
                )
            fig.for_each_annotation(lambda a: a.update(text=a.text))
            st.plotly_chart(fig, use_container_width=True)

            fig_box = px.box(data_dec, x="decile", y=target, category_orders={"decile": labels})
            st.plotly_chart(fig_box, use_container_width=True)

        elif is_numeric_dtype(data[feature]):
            df_num = data[[feature, target]].dropna()
            color_arg = "Кластер" if colorize and "Кластер" in data.columns else None
            with col1:
                st.plotly_chart(px.scatter(df_num, x=feature, y=target, color=color_arg), use_container_width=True)
            with col2:
                st.plotly_chart(px.histogram(x=df_num[feature], nbins=bins, histnorm=histnorm), use_container_width=True)
            st.plotly_chart(px.box(y=df_num[feature]), use_container_width=True)

        else:
            vals = data[feature].astype(str).value_counts()
            top_vals = vals.nlargest(10).index.tolist() if len(vals) > 10 else vals.index.tolist()
            grp = data[feature].astype(str).where(data[feature].astype(str).isin(top_vals), "Прочие")
            cats = sorted(top_vals) + (["Прочие"] if len(vals) > 10 else [])
            with col1:
                bar_df = data.groupby(grp)[target].mean().reindex(cats).reset_index()
                st.plotly_chart(
                    px.bar(bar_df, x=grp.name or feature, y=target, category_orders={grp.name or feature: cats}),
                    use_container_width=True,
                )
            with col2:
                cnt_df = grp.value_counts().reindex(cats).reset_index()
                cnt_df.columns = [grp.name or feature, "count"]
                st.plotly_chart(
                    px.bar(cnt_df, x=grp.name or feature, y="count", category_orders={grp.name or feature: cats}),
                    use_container_width=True,
                )
            fig = px.histogram(
                data.assign(grp=grp),
                x=target,
                facet_col="grp",
                facet_col_wrap=4,
                nbins=bins,
                histnorm=histnorm,
                category_orders={"grp": cats},
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(
                px.box(
                    data.assign(grp=grp),
                    x="grp",
                    y=target,
                    category_orders={"grp": cats},
                ),
                use_container_width=True,
            )
            if selected_clusters:
                st.subheader("Распределение по кластерам")
                df_grp = data.assign(grp=grp)
                fig_cluster_hist = px.histogram(
                    df_grp,
                    x="grp",
                    facet_col="Кластер",
                    facet_col_wrap=3,
                    histnorm=histnorm,
                    category_orders={"Кластер": selected_clusters, "grp": cats},
                )
                fig_cluster_hist.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(fig_cluster_hist, use_container_width=True)

                st.subheader("Boxplot по кластерам")
                fig_cluster_box = px.box(
                    df_grp,
                    x="grp",
                    y=target,
                    facet_col="Кластер",
                    facet_col_wrap=3,
                    category_orders={"Кластер": selected_clusters, "grp": cats},
                )
                fig_cluster_box.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(fig_cluster_box, use_container_width=True)


if __name__ == "__main__":
    main()