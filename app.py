import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

@st.cache_data
def load_data(path):
    df = pd.read_excel(path, header=3)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in ["Дата регистрации", "Дата рождения"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    if set(["Дата регистрации", "Дата рождения"]).issubset(df.columns):
        df["Возраст"] = ((df["Дата регистрации"] - df["Дата рождения"]).dt.days / 365.25)\
                            .floordiv(1).astype('Int64')
    if "Сумма договора" in df.columns:
        low, high = df["Сумма договора"].quantile([0.01, 0.99])
        df["Сумма договора"] = df["Сумма договора"].clip(low, high)
    if set(["Сумма договора", "Дата регистрации"]).issubset(df.columns):
        ir = get_interest_rate(df, "Сумма договора", "Дата регистрации")
        df["Сумма договора (disc)"] = discounting(df, ir, "Сумма договора", "Дата регистрации")
    if "Email" in df.columns:
        df["Email"] = df["Email"].str.split("@").str[-1]
    if "Контрагент" in df.columns:
        df["Пол"] = df["Контрагент"].apply(extract_gender)
    return df


def get_interest_rate(deals, target_name, date_col, is_ml=True):
    df = deals.sort_values(date_col)
    monthly = df.set_index(date_col)[target_name]    \
                .resample("M").mean().to_period("M")
    pct = monthly.pct_change().shift() if is_ml else monthly.pct_change()
    return (1 + pct.fillna(0)).cumprod()


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
    df = load_data("Список.xlsx")
    targets = [c for c in ["Сумма договора", "Сумма договора (disc)", "Площадь договора"]
               if c in df.columns]
    all_feats = [c for c in [
        "Вид помещения", "Контрагент", "Дата рождения", "Телефон", "Email",
        "Объект строительства", "Проект", "Текущий регион", "Дата регистрации",
        "Типология", "Возраст"
    ] if c in df.columns]
    analysis_feats = [c for c in all_feats if c not in ["Телефон", "Вид помещения"]]

    section = st.sidebar.radio("Раздел", ["Обзор признаков", "Анализ данных"], key="sec")

    if section == "Обзор признаков":
        st.header("Обзор признаков и целевых метрик")
        bins = st.sidebar.slider("Количество бинов", 5, 100, 30, 5)
        normalize = st.sidebar.checkbox("Нормализовать распределения")
        histnorm = "percent" if normalize else None

        for col in all_feats + targets:
            st.subheader(col)
            c1, c2 = st.columns(2)
            if col == "Телефон":
                vals = df[col].astype(str).str.replace(r"\D", "", regex=True).str[:3]
                counts = vals.value_counts()
                top = counts.index if len(counts) <= 10 else counts.nlargest(10).index
                grp = vals.where(vals.isin(top), "Прочие") if len(counts) > 10 else vals
                cnt = grp.value_counts().reindex(list(top) + (["Прочие"] if len(counts) > 10 else []))
                with c1:
                    st.write(cnt)
                with c2:
                    fig = px.bar(x=cnt.index, y=cnt.values)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
            elif is_numeric_dtype(df[col]):
                x = df[col].dropna()
                with c1:
                    st.write(x.describe())
                with c2:
                    fig = px.histogram(x=x, nbins=bins, histnorm=histnorm)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
            elif is_datetime64_any_dtype(df[col]):
                dates = df[col].dropna()
                months = dates.dt.to_period('M').dt.to_timestamp()
                with c1:
                    st.write(dates.describe(datetime_is_numeric=True))
                with c2:
                    fig = px.histogram(x=months, nbins=bins, histnorm=histnorm)
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title='Месяц'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                vals = df[col].astype(str).value_counts()
                vals = vals.sort_index()
                with c1:
                    st.write(vals)
                with c2:
                    fig = px.bar(x=vals.index, y=vals.values)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.header("Анализ данных")
        target = st.sidebar.selectbox("Целевая переменная", targets)
        feature = st.sidebar.selectbox("Признак", analysis_feats)
        regions = sorted(df.get("Текущий регион", pd.Series()).dropna().unique())
        cities = st.sidebar.multiselect("Регион", ["Все"] + regions, default=["Все"])
        bins = st.sidebar.slider("Количество бинов", 5, 100, 30, 5)
        normalize = st.sidebar.checkbox("Нормализовать распределения")
        histnorm = "percent" if normalize else None

        data = df.copy()
        if "Все" not in cities:
            data = data[data["Текущий регион"].isin(cities)]
        if "Вид помещения" in data.columns:
            data = data[data["Вид помещения"] == "Квартира"]

        remove_feature = st.sidebar.checkbox("Удалить выбросы по признаку")
        remove_target = st.sidebar.checkbox("Удалить выбросы по таргету")
        if remove_feature:
            if is_datetime64_any_dtype(data[feature]):
                dt_int = data[feature].dropna().astype('int64')
                low_i, high_i = np.quantile(dt_int, [0.01, 0.99])
                low_dt = pd.to_datetime(low_i)
                high_dt = pd.to_datetime(high_i)
                data = data[(data[feature] >= low_dt) & (data[feature] <= high_dt)]
            elif is_numeric_dtype(data[feature]):
                vals = data[feature].dropna()
                low_f, high_f = vals.quantile([0.01, 0.99])
                data = data[(data[feature] >= low_f) & (data[feature] <= high_f)]
        if remove_target and is_numeric_dtype(data[target]):
            tvals = data[target].dropna()
            low_t, high_t = tvals.quantile([0.01, 0.99])
            data = data[(data[target] >= low_t) & (data[target] <= high_t)]

        col1, col2 = st.columns(2)

        if is_datetime64_any_dtype(data[feature]) or feature == "Возраст":
            if is_datetime64_any_dtype(data[feature]):
                x_val = data[feature].dropna().dt.to_period('M').dt.to_timestamp()
            else:
                x_val = data[feature].dropna()
            df_raw = data.assign(x=x_val).dropna(subset=['x', target])

            st.subheader("Scatter")
            st.plotly_chart(
                px.scatter(df_raw, x='x', y=target)
                  .update_layout(xaxis_title=feature, yaxis_title=target),
                use_container_width=True
            )

            st.subheader("Копула")
            x_q = x_val.rank(method='average', pct=True)
            t_q = data[target].rank(method='average', pct=True)
            cop = pd.DataFrame({'x_q': x_q, 't_q': t_q}).dropna()

            fig = px.scatter(
                cop,
                x='x_q',
                y='t_q',
                labels={
                    'x_q': f'Квантиль {feature} (pct)',
                    't_q': f'Квантиль {target} (pct)'
                }
            )
            fig.update_layout(
                xaxis=dict(range=[0,1]),
                yaxis=dict(range=[0,1]),
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_df = df_raw.groupby('x')[target].mean().reset_index()
            with col1:
                st.subheader("Среднее по месяцам")
                st.plotly_chart(
                    px.scatter(avg_df, x='x', y=target)
                      .update_layout(xaxis_title=feature, yaxis_title=target),
                    use_container_width=True
                )
            with col2:
                st.subheader("Среднее по году")
                st.plotly_chart(
                    px.line(avg_df, x='x', y=target)
                      .update_layout(xaxis_title=feature, yaxis_title=target),
                    use_container_width=True
                )

            st.subheader(f"Распределение {feature}")
            st.plotly_chart(
                px.histogram(x=x_val, nbins=bins)
                  .update_layout(xaxis_title=feature, yaxis_title='count'),
                use_container_width=True
            )

            edges = pd.qcut(x_val, 10, retbins=True, duplicates='drop')[1]
            if feature != "Возраст":
                edges = pd.to_datetime(edges)
            labels = []
            for i in range(len(edges)-1):
                l, r = edges[i], edges[i+1]
                labels.append(
                    f"{int(l)}-{int(r)}" if feature=='Возраст' else f"{l.strftime('%Y-%m')}-{r.strftime('%Y-%m')}"
                )
            data_dec = data.assign(x=x_val)
            data_dec['decile'] = pd.cut(
                x_val,
                bins=edges,
                labels=labels,
                include_lowest=True
            ).dropna()
            st.subheader("Таргет по интервалам")
            fig = px.histogram(
                data_dec,
                x=target,
                facet_col='decile',
                facet_col_wrap=5,
                nbins=bins,
                histnorm=histnorm,
                category_orders={'decile': labels}
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Боксплот таргета по интервалам")
            fig_box = px.box(
                data_dec,
                x='decile',
                y=target,
                category_orders={'decile': labels}
            )
            st.plotly_chart(fig_box, use_container_width=True)

        elif is_numeric_dtype(data[feature]):
            df_num = data[[feature, target]].dropna()
            with col1:
                st.plotly_chart(px.scatter(df_num, x=feature, y=target), use_container_width=True)
            with col2:
                st.plotly_chart(
                    px.histogram(x=df_num[feature], nbins=bins, histnorm=histnorm),
                    use_container_width=True
                )
            st.plotly_chart(px.box(y=df_num[feature]), use_container_width=True)

        else:
            vals = data[feature].astype(str).value_counts()
            if len(vals) > 10:
                top_vals = vals.nlargest(10).index.tolist()
            else:
                top_vals = vals.index.tolist()
            sorted_vals = sorted(top_vals)
            grp = data[feature].astype(str).where(data[feature].astype(str).isin(top_vals), 'Прочие')
            cats = sorted_vals + (['Прочие'] if len(vals) > 10 else [])

            with col1:
                bar_df = data.groupby(grp)[target].mean().reindex(cats).reset_index()
                st.plotly_chart(
                    px.bar(
                        bar_df,
                        x=grp.name or feature,
                        y=target,
                        category_orders={grp.name or feature: cats}
                    ),
                    use_container_width=True
                )
            with col2:
                cnt_df = grp.value_counts().reindex(cats).reset_index()
                cnt_df.columns = [grp.name or feature, 'count']
                st.plotly_chart(
                    px.bar(
                        cnt_df,
                        x=grp.name or feature,
                        y='count',
                        category_orders={grp.name or feature: cats}
                    ),
                    use_container_width=True
                )
            fig = px.histogram(
                data.assign(grp=grp),
                x=target,
                facet_col='grp',
                facet_col_wrap=4,
                nbins=bins,
                histnorm=histnorm,
                category_orders={'grp': cats}
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(
                px.box(
                    data.assign(grp=grp),
                    x='grp',
                    y=target,
                    category_orders={'grp': cats}
                ),
                use_container_width=True
            )

if __name__ == "__main__":
    main()