import json
from pathlib import Path

import joblib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn import metrics
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино',
}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def batch_predictions(file, model, features, threshold):
    try:
        data = pd.read_csv(file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError('Не удалось прочитать CSV. Проверьте кодировку UTF-8 и разделитель запятую.') from error

    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('В файле нет строк с винами.')

    wines = data.loc[:, features].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(wines.to_numpy(dtype=float)).all():
        raise ValueError('Заполните все признаки конечными числами.')
    if not wines['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red должны быть 0 для белого вина и 1 для красного.')

    scores = model.predict_proba(wines)[:, 1]
    return wines.assign(
        probability=scores,
        decision=np.where(scores >= threshold, 'Добавить в подборку', 'Пропустить'),
    )


st.set_page_config(page_title='Подборка вин', layout='wide')
st.title('Подборка вин')

model = load_model()
test = load_test()
features = model.feature_names_in_.tolist()
X_test = test[features]
y_test = test['target']
test_scores = test_proba(model, X_test)

default_threshold = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))['threshold']
threshold = st.sidebar.slider('Порог рекомендации', 0.0, 1.0, default_threshold, 0.001, format='%.4f')
wine_tab, metrics_tab, csv_tab, weights_tab = st.tabs([
    'Оценка вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель',
])

with wine_tab:
    wine_type = st.radio('Цвет вина', ['Белое', 'Красное'], horizontal=True)
    wine = {'is_red': int(wine_type == 'Красное')}
    columns = st.columns(3)
    numeric_features = [feature for feature in features if feature != 'is_red']
    for index, feature in enumerate(numeric_features):
        values = X_test[feature]
        step = 0.00001 if feature == 'density' else 0.001
        wine[feature] = columns[index % len(columns)].number_input(
            FEATURE_LABELS[feature],
            min_value=float(values.min()),
            max_value=float(values.max()),
            value=float(values.median()),
            step=step,
            format='%.5f' if feature == 'density' else '%.3f',
        )

    wine_frame = pd.DataFrame([wine], columns=features)
    probability = model.predict_proba(wine_frame)[0, 1]
    left, right = st.columns(2)
    left.metric('Вероятность хорошего вина', f'{probability:.1%}')
    right.metric('Решение', 'Добавить в подборку' if probability >= threshold else 'Пропустить')

with metrics_tab:
    predicted = test_scores >= threshold
    matrix = metrics.confusion_matrix(y_test, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    precision = metrics.precision_score(y_test, predicted, zero_division=0)
    recall = metrics.recall_score(y_test, predicted, zero_division=0)
    f1 = metrics.f1_score(y_test, predicted, zero_division=0)
    for column, (name, value) in zip(st.columns(3), [('Precision', precision), ('Recall', recall), ('F1', f1)]):
        column.metric(name, f'{value:.3f}')

    st.subheader('Матрица ошибок')
    st.dataframe(pd.DataFrame(
        matrix,
        index=['На деле класс 0', 'На деле класс 1'],
        columns=['Пропустить', 'Добавить в подборку'],
    ))

    fpr, tpr, _ = metrics.roc_curve(y_test, test_scores)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, test_scores)
    fig_roc = Figure(figsize=(5, 4), layout='tight')
    ax_roc = fig_roc.subplots()
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1], ':', color='gray')
    ax_roc.scatter(fp / (fp + tn), recall, color='tab:orange', label='Текущий порог', zorder=3)
    ax_roc.set(xlabel='Доля ложных срабатываний', ylabel='Recall', title='ROC кривая')
    ax_roc.legend()

    fig_pr = Figure(figsize=(5, 4), layout='tight')
    ax_pr = fig_pr.subplots()
    ax_pr.step(pr_recall, pr_precision, where='post')
    ax_pr.axhline(y_test.mean(), linestyle=':', color='gray')
    point_precision = precision if tp + fp else 1
    ax_pr.scatter(recall, point_precision, color='tab:orange', label='Текущий порог', zorder=3)
    ax_pr.set(xlabel='Recall', ylabel='Precision', title='PR кривая')
    ax_pr.legend()

    left, right = st.columns(2)
    left.pyplot(fig_roc)
    right.pyplot(fig_pr)
    if tp + fp == 0:
        st.caption('При пустой подборке точка на PR кривой показана в начале при Recall = 0 и Precision = 1 по соглашению для этой кривой.')
    st.write('При повышении порога в подборку попадает меньше вин и можно пропустить больше хороших. '
             'При снижении порога подборка расширяется, но вместе с хорошими винами в неё могут попасть посредственные.')

with csv_tab:
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    st.caption('Столбцы: ' + ', '.join(features))
    if uploaded is not None:
        try:
            result = batch_predictions(uploaded, model, features, threshold)
        except ValueError as error:
            st.error(str(error))
        else:
            st.dataframe(result)
            st.download_button(
                'Скачать предсказания', result.to_csv(index=False).encode('utf-8-sig'),
                file_name='predictions.csv', mime='text/csv',
            )

with weights_tab:
    coefficients = pd.Series(model[-1].coef_[0], index=features).sort_values()
    labels = [FEATURE_LABELS[feature] for feature in coefficients.index]
    colors = np.where(coefficients > 0, 'tab:blue', 'tab:orange')
    figure = Figure(figsize=(8, 5), layout='tight')
    axis = figure.subplots()
    axis.barh(labels, coefficients, color=colors)
    axis.axvline(0, color='gray', linewidth=1)
    axis.set_xlabel('Коэффициент для масштабированного признака')
    st.pyplot(figure)
    st.write(f'Сильнее всего за хорошее вино: {FEATURE_LABELS[coefficients.idxmax()]}.')
    st.write(f'Сильнее всего против: {FEATURE_LABELS[coefficients.idxmin()]}.')
