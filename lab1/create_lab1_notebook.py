from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path("lab1_heart_disease_preprocessing.ipynb")


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


nb = nbf.v4.new_notebook()

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

nb["cells"] = [
    md(
        """
        # Практическая работа №1
        ## Предобработка данных интеллектуальных моделей

        **Дисциплина:** Математика в программировании, часть 2/2.

        **Тема работы:** предобработка табличных медицинских данных для задачи диагностики наличия сердечно-сосудистого заболевания.

        **Выбранная задача для последующих практик:** бинарная классификация: по клиническим признакам пациента определить, есть ли у пациента признаки заболевания сердца.
        """
    ),
    md(
        """
        ## 1. Предметная область и постановка задачи

        Предметная область: медицинская диагностика сердечно-сосудистых заболеваний.

        В исходном наборе данных целевая переменная `num` принимает значения:

        - `0` - заболевание не выявлено;
        - `1`, `2`, `3`, `4` - заболевание выявлено с разной степенью выраженности.

        Для базовой интеллектуальной модели целевая переменная преобразуется в бинарную:

        - `0` - заболевание сердца отсутствует;
        - `1` - заболевание сердца присутствует.

        Такая постановка соответствует дискриминативной задаче бинарной классификации.
        """
    ),
    md(
        """
        ## 2. Сбор данных и описание источника

        Используется набор данных **UCI Heart Disease** из локальной папки `heart+disease`.

        Набор содержит клинические данные, собранные в нескольких медицинских учреждениях:

        - Cleveland Clinic Foundation;
        - Hungarian Institute of Cardiology, Budapest;
        - V.A. Medical Center, Long Beach;
        - University Hospital, Zurich, Switzerland.

        В полном наборе описано 76 атрибутов, но в большинстве экспериментов применялись 14 основных признаков. В работе используются файлы `processed.*.data`, потому что они уже приведены к этой общей 14-признаковой схеме.

        Особенность качества данных: в файле `WARNING` указано, что сырой `cleveland.data` был поврежден, а `processed.cleveland.data` пригоден для использования. Поэтому для обработки берутся именно processed-файлы.
        """
    ),
    code(
        """
        from pathlib import Path
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from sklearn.compose import ColumnTransformer
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.manifold import TSNE
        from sklearn.metrics import ConfusionMatrixDisplay, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", palette="Set2")
        pd.set_option("display.max_columns", 80)
        pd.set_option("display.width", 140)

        DATA_DIR = Path("heart+disease")
        """
    ),
    md(
        """
        ## 3. Загрузка данных

        Все processed-файлы имеют одинаковый порядок 14 колонок. Имена колонок взяты из `heart-disease.names`.
        """
    ),
    code(
        """
        columns = [
            "age",       # возраст
            "sex",       # пол: 1 - male, 0 - female
            "cp",        # тип боли в груди
            "trestbps",  # давление в покое
            "chol",      # холестерин
            "fbs",       # сахар натощак > 120 mg/dl
            "restecg",   # ЭКГ в покое
            "thalach",   # максимальная ЧСС
            "exang",     # стенокардия при нагрузке
            "oldpeak",   # депрессия ST
            "slope",     # наклон сегмента ST
            "ca",        # число крупных сосудов
            "thal",      # результат thal-теста
            "num",       # целевая переменная
        ]

        source_files = {
            "Cleveland": "processed.cleveland.data",
            "Hungarian": "processed.hungarian.data",
            "Switzerland": "processed.switzerland.data",
            "Long Beach VA": "processed.va.data",
        }

        frames = []
        for source, filename in source_files.items():
            part = pd.read_csv(
                DATA_DIR / filename,
                header=None,
                names=columns,
                na_values="?",
            )
            part["source"] = source
            frames.append(part)

        raw = pd.concat(frames, ignore_index=True)
        raw.head()
        """
    ),
    code(
        """
        print(f"Размер объединенного набора: {raw.shape[0]} строк, {raw.shape[1]} колонок")
        display(raw.groupby("source").size().rename("rows").to_frame())
        display(raw.dtypes.to_frame("dtype").T)
        """
    ),
    md(
        """
        ## 4. Анализ качества данных

        Проверяются:

        - пропущенные значения;
        - дубликаты;
        - распределение целевой переменной;
        - невозможные или подозрительные числовые значения.
        """
    ),
    code(
        """
        missing = (
            raw.isna()
            .sum()
            .to_frame("missing_count")
            .assign(missing_percent=lambda x: (x["missing_count"] / len(raw) * 100).round(2))
            .sort_values("missing_percent", ascending=False)
        )
        display(missing)

        plt.figure(figsize=(10, 4))
        sns.barplot(data=missing.reset_index(), x="index", y="missing_percent")
        plt.xticks(rotation=45, ha="right")
        plt.title("Доля пропусков по признакам")
        plt.xlabel("Признак")
        plt.ylabel("Пропуски, %")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        duplicate_count = raw.duplicated().sum()
        print(f"Количество полных дубликатов: {duplicate_count}")

        target_distribution = raw["num"].value_counts().sort_index().rename_axis("num").to_frame("count")
        target_distribution["percent"] = (target_distribution["count"] / len(raw) * 100).round(2)
        display(target_distribution)

        plt.figure(figsize=(7, 4))
        sns.countplot(data=raw, x="num", hue="source")
        plt.title("Распределение исходной целевой переменной по источникам")
        plt.xlabel("num")
        plt.ylabel("Количество")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        numeric_medical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        display(raw[numeric_medical_cols].describe().T)

        zero_checks = pd.DataFrame({
            "zero_count": {
                col: int((raw[col] == 0).sum())
                for col in ["trestbps", "chol", "thalach"]
            }
        })
        display(zero_checks)
        """
    ),
    md(
        """
        Вывод по качеству данных:

        - пропуски представлены символом `?` и при загрузке заменены на `NaN`;
        - признак `ca` отсутствует более чем у 60% записей, поэтому его рационально удалить;
        - признак `thal` отсутствует более чем у половины записей, но он оставлен как диагностически значимый категориальный признак и будет заполнен модой;
        - для `chol` встречается много нулевых значений; для холестерина это физически неправдоподобно, поэтому такие значения рассматриваются как скрытые пропуски;
        - для `trestbps` найдено нулевое значение; оно также заменяется на пропуск;
        - полные дубликаты можно удалить как часть очистки данных.
        """
    ),
    md(
        """
        ## 5. Предобработка данных

        Выполняемые шаги:

        1. удаление полных дубликатов;
        2. преобразование целевой переменной `num` в бинарный `target`;
        3. замена скрытых пропусков в `chol` и `trestbps`;
        4. удаление признака `ca` из-за слишком большой доли пропусков;
        5. добавление индикаторов пропусков для признаков, где они были обнаружены;
        6. заполнение числовых признаков медианой;
        7. заполнение категориальных признаков модой;
        8. стандартизация числовых признаков;
        9. one-hot encoding категориальных признаков.
        """
    ),
    code(
        """
        df = raw.copy()
        df = df.drop_duplicates().reset_index(drop=True)

        df["target"] = (df["num"] > 0).astype(int)

        # Нули в этих медицинских признаках считаем скрытыми пропусками.
        df.loc[df["chol"] == 0, "chol"] = np.nan
        df.loc[df["trestbps"] == 0, "trestbps"] = np.nan

        initial_features = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        missing_rate_after_zero_fix = df[initial_features].isna().mean().sort_values(ascending=False)
        high_missing_cols = missing_rate_after_zero_fix[missing_rate_after_zero_fix > 0.60].index.tolist()
        print("Признаки, удаляемые из-за доли пропусков > 60%:", high_missing_cols)

        model_features = [col for col in initial_features if col not in high_missing_cols]
        cols_with_missing = [col for col in model_features if df[col].isna().any()]

        for col in cols_with_missing:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

        indicator_features = [f"{col}_was_missing" for col in cols_with_missing]

        numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

        X = df[model_features + indicator_features]
        y = df["target"]

        print(f"После удаления дубликатов: {df.shape[0]} строк")
        print(f"Количество признаков до one-hot encoding: {X.shape[1]}")
        display(y.value_counts().sort_index().rename({0: "нет болезни", 1: "есть болезнь"}).to_frame("count"))
        """
    ),
    code(
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                ("missing_flag", "passthrough", indicator_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        preprocessor.set_output(transform="pandas")
        X_prepared = preprocessor.fit_transform(X)

        prepared = X_prepared.copy()
        prepared["target"] = y.to_numpy()
        prepared["severity_num"] = df["num"].to_numpy()
        prepared["source"] = df["source"].to_numpy()

        output_path = Path("heart_disease_preprocessed.csv")
        prepared.to_csv(output_path, index=False)

        print(f"Итоговый размер матрицы признаков: {X_prepared.shape}")
        print(f"Сохранено: {output_path}")
        display(prepared.head())
        """
    ),
    md(
        """
        ## 6. Проверка пригодности предобработки на простой модели

        Этот шаг не является основной целью практической работы, но он показывает, что после предобработки данные можно передать в интеллектуальную модель без ошибок формата, пропусков и категориальных строк.
        """
    ),
    code(
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred, target_names=["нет болезни", "есть болезнь"]))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["нет болезни", "есть болезнь"],
            cmap="Blues",
        )
        plt.title("Матрица ошибок простой логистической регрессии")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 7. Аугментация табличных данных

        Для табличных данных применяется простая статистическая аугментация:

        - выбираются записи меньшего класса;
        - записи сэмплируются с возвращением;
        - к непрерывным признакам добавляется небольшой нормальный шум, пропорциональный стандартному отклонению признака;
        - категориальные признаки сохраняются без изменения;
        - значения ограничиваются наблюдаемыми диапазонами, чтобы не получать явно невозможные величины.

        Такая аугментация не заменяет реальный медицинский сбор данных, но демонстрирует подход генерации типичных статистически близких табличных записей.
        """
    ),
    code(
        """
        def make_statistical_augmentation(data, target_col, numeric_cols, categorical_cols, random_state=42):
            rng = np.random.default_rng(random_state)
            base = data[numeric_cols + categorical_cols + [target_col]].copy()

            for col in numeric_cols:
                base[col] = base[col].fillna(base[col].median())
            for col in categorical_cols:
                base[col] = base[col].fillna(base[col].mode(dropna=True).iloc[0])

            counts = base[target_col].value_counts()
            majority_size = counts.max()
            augmented_parts = []

            for cls, count in counts.items():
                need = int(majority_size - count)
                if need <= 0:
                    continue

                cls_rows = base[base[target_col] == cls]
                sampled = cls_rows.sample(n=need, replace=True, random_state=random_state).copy()

                for col in numeric_cols:
                    std = cls_rows[col].std()
                    noise_scale = 0.03 * std if pd.notna(std) and std > 0 else 0
                    sampled[col] = sampled[col] + rng.normal(0, noise_scale, size=need)

                    low, high = base[col].quantile([0.01, 0.99])
                    sampled[col] = sampled[col].clip(low, high)

                sampled["is_augmented"] = 1
                augmented_parts.append(sampled)

            original = base.copy()
            original["is_augmented"] = 0

            if augmented_parts:
                return pd.concat([original] + augmented_parts, ignore_index=True)
            return original


        augmented = make_statistical_augmentation(
            df,
            target_col="target",
            numeric_cols=numeric_features,
            categorical_cols=categorical_features,
            random_state=42,
        )

        augmented_path = Path("heart_disease_augmented.csv")
        augmented.to_csv(augmented_path, index=False)

        display(pd.crosstab(augmented["target"], augmented["is_augmented"], margins=True))
        print(f"Сохранено: {augmented_path}")
        """
    ),
    md(
        """
        ## 8. Визуализация данных

        Для визуального анализа многомерных данных используются:

        - графики по отдельным клиническим признакам;
        - PCA для линейного понижения размерности;
        - t-SNE для нелинейной визуализации локальной структуры.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        sns.histplot(data=df, x="age", hue="target", kde=True, ax=axes[0])
        axes[0].set_title("Возраст по классам")

        sns.scatterplot(data=df, x="age", y="thalach", hue="target", style="target", ax=axes[1])
        axes[1].set_title("Возраст и максимальная ЧСС")

        sns.scatterplot(data=df, x="chol", y="trestbps", hue="target", style="target", ax=axes[2])
        axes[2].set_title("Холестерин и давление")

        for ax in axes:
            ax.set_xlabel(ax.get_xlabel())
            ax.set_ylabel(ax.get_ylabel())

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(X_prepared)
        pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
        pca_df["target"] = y.map({0: "нет болезни", 1: "есть болезнь"}).to_numpy()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="target", alpha=0.75)
        plt.title(
            "PCA-визуализация предобработанных данных "
            f"(объяснено {pca.explained_variance_ratio_.sum() * 100:.1f}% дисперсии)"
        )
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        # Перед t-SNE уменьшаем размерность PCA до 30 компонент для ускорения и снижения шума.
        pca_for_tsne = PCA(n_components=min(30, X_prepared.shape[1]), random_state=42)
        tsne_input = pca_for_tsne.fit_transform(X_prepared)

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=42,
        )
        tsne_coords = tsne.fit_transform(tsne_input)

        tsne_df = pd.DataFrame(tsne_coords, columns=["TSNE1", "TSNE2"])
        tsne_df["target"] = y.map({0: "нет болезни", 1: "есть болезнь"}).to_numpy()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="target", alpha=0.75)
        plt.title("t-SNE-визуализация предобработанных данных")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 9. Итоги

        В работе выполнен полный цикл подготовки табличного медицинского набора данных:

        - определена предметная область и задача бинарной классификации;
        - выбран и описан источник данных UCI Heart Disease;
        - объединены данные из четырех processed-файлов;
        - выполнен анализ качества данных;
        - обработаны явные пропуски `?` и скрытые пропуски в виде нулевых медицинских значений;
        - удален признак с чрезмерной долей пропусков;
        - добавлены индикаторы пропусков;
        - числовые признаки заполнены медианой и стандартизированы;
        - категориальные признаки заполнены модой и закодированы one-hot encoding;
        - выполнена статистическая аугментация табличных данных;
        - построены визуализации признаков, PCA и t-SNE;
        - проверено, что предобработанные данные подходят для обучения простой интеллектуальной модели.

        Полученные файлы:

        - `heart_disease_preprocessed.csv` - предобработанная матрица признаков;
        - `heart_disease_augmented.csv` - пример аугментированного набора данных.
        """
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
