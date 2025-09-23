import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
warnings.filterwarnings('ignore')

try:
    import cudf
    import cuml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class OptimizedTextAwarePipeline:
    """
    Полный оптимизированный пайплайн с поэтапным уменьшением выборки,
    правильной обработкой признаков и визуализациями
    """
    
    def __init__(self, target='resolution'):
        self.target = target
        self.col_drop = ['id', 'ItemID', 'SellerID']
        self.hard_threshold_low = 0.2
        self.hard_threshold_high = 0.8
        
        self.text_columns = []
        self.models = {}
        self.features = {}
        self.label_encoders = {}  # Для кодирования категориальных признаков
        
        # GPU detection
        self.gpu_available = GPU_AVAILABLE
        self.device_type = "GPU" if self.gpu_available else "CPU"
        
        # Настройка обработки текста для CatBoost
        self.text_processing_params = {
            'tokenizers': [{'tokenizer_id': 'Space', 'separator_type': 'ByDelimiter', 'delimiter': ' '}],
            'dictionaries': [{'dictionary_id': 'Word', 'max_dictionary_size': 50000}],
            'feature_calcers': [{'calcer_id': 'BoW', 'top_tokens_count': 1000}]
        }
        
        # Оптимизированные параметры CatBoost
        self.base_catboost_params = {
            'random_seed': 255,
            'task_type': self.device_type,
            'verbose': False,
            'eval_metric': 'F1',
            'loss_function': 'Logloss',
            'use_best_model': True,
            'early_stopping_rounds': 30,
            'thread_count': -1,
            'allow_writing_files': False,
            'text_processing': self.text_processing_params
        }
        
        if self.gpu_available:
            self.base_catboost_params.update({
                'gpu_ram_part': 0.8,
                'used_ram_limit': '8gb'
            })
    
    def identify_text_columns(self, df: pl.DataFrame):
        """Определяем текстовые колонки"""
        self.text_columns = [col for col, dtype in df.schema.items() 
                           if dtype == pl.String and col not in self.col_drop + [self.target]]
        print(f"Найдено текстовых колонок: {len(self.text_columns)}")
        return self.text_columns
    
    def advanced_feature_engineering(self, df: pl.DataFrame, stage: str) -> pl.DataFrame:
        """
        Правильная генерация признаков с разделением на continuous/categorical
        """
        print(f"Feature engineering для стадии {stage}...")
        
        # Получаем числовые колонки
        numeric_cols = []
        for col, dtype in df.schema.items():
            if col not in self.col_drop + [self.target] + self.text_columns:
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    numeric_cols.append(col)
        
        # ПРАВИЛЬНО: Разделяем на continuous и categorical
        continuous_cols = []
        categorical_cols = []
        
        for col in numeric_cols:
            if col in df.columns:
                unique_count = df[col].n_unique()
                if unique_count <= 50:  # Категориальные
                    categorical_cols.append(col)
                else:  # Непрерывные
                    continuous_cols.append(col)
        
        print(f"Continuous: {len(continuous_cols)}, Categorical: {len(categorical_cols)}")
        
        expressions = []
        
        # 1. Трансформации ТОЛЬКО для continuous признаков
        for col in continuous_cols[:15]:
            if col in df.columns:
                expressions.extend([
                    pl.when(pl.col(col) > 0).then(pl.col(col).log()).otherwise(0).alias(f"{col}_log"),
                    pl.when(pl.col(col) >= 0).then(pl.col(col).sqrt()).otherwise(0).alias(f"{col}_sqrt"),
                    (pl.col(col) ** 2).alias(f"{col}_sq"),
                    pl.when(pl.col(col) != 0).then(1.0 / pl.col(col)).otherwise(0).alias(f"{col}_inv")
                ])
        
        # 2. Парные взаимодействия для continuous
        if len(continuous_cols) >= 2:
            top_continuous = continuous_cols[:6]  # Ограничиваем для производительности
            for i in range(len(top_continuous)):
                for j in range(i+1, len(top_continuous)):
                    col1, col2 = top_continuous[i], top_continuous[j]
                    if col1 in df.columns and col2 in df.columns:
                        expressions.extend([
                            pl.when(pl.col(col2) != 0).then(pl.col(col1) / pl.col(col2)).otherwise(0).alias(f"{col1}_{col2}_ratio"),
                            (pl.col(col1) * pl.col(col2)).alias(f"{col1}_{col2}_prod"),
                            (pl.col(col1) - pl.col(col2)).alias(f"{col1}_{col2}_diff")
                        ])
        
        # 3. Percentile-based features для continuous
        for col in continuous_cols[:10]:
            if col in df.columns:
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                median = df[col].median()
                
                expressions.extend([
                    (pl.col(col) > q75).cast(pl.Int8).alias(f"{col}_q75_flag"),
                    (pl.col(col) < q25).cast(pl.Int8).alias(f"{col}_q25_flag"),
                    (pl.col(col) > median).cast(pl.Int8).alias(f"{col}_median_flag")
                ])
        
        # Применяем трансформации батчами
        if expressions:
            batch_size = 50
            for i in range(0, len(expressions), batch_size):
                batch = expressions[i:i+batch_size]
                try:
                    df = df.with_columns(batch)
                except Exception as e:
                    print(f"Ошибка в batch {i}: {e}")
                    continue
        
        print(f"Сгенерировано {len(expressions)} новых признаков для {stage}")
        return df
    
    def improved_feature_selection(self, df_pandas: pd.DataFrame, stage: str, max_features: int = 30) -> list:
        """
        Улучшенный отбор признаков без PHIK
        """
        print(f"Feature selection для {stage} (макс. {max_features} признаков)...")
        
        # Получаем все признаки кроме служебных
        feature_cols = [col for col in df_pandas.columns 
                       if col not in self.col_drop + [self.target]]
        
        if len(feature_cols) < 2:
            return feature_cols
        
        # Разделяем на числовые и текстовые
        numeric_features = []
        text_features = []
        
        for col in feature_cols:
            if col in self.text_columns:
                text_features.append(col)
            else:
                try:
                    # Проверяем, является ли колонка числовой
                    pd.to_numeric(df_pandas[col], errors='raise')
                    numeric_features.append(col)
                except (ValueError, TypeError):
                    # Если не числовая, кодируем как категориальную
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df_pandas[col] = self.label_encoders[col].fit_transform(df_pandas[col].fillna('missing'))
                    numeric_features.append(col)
        
        selected_features = []
        
        # Mutual information для числовых признаков
        if numeric_features and len(numeric_features) > 0:
            X_numeric = df_pandas[numeric_features].fillna(0)
            y = df_pandas[self.target]
            
            try:
                mi_selector = SelectKBest(mutual_info_classif, k=min(max_features-5, len(numeric_features)))
                mi_selector.fit(X_numeric, y)
                selected_numeric = [numeric_features[i] for i in mi_selector.get_support(indices=True)]
                selected_features.extend(selected_numeric)
            except Exception as e:
                print(f"MI selection failed: {e}, using correlation")
                # Fallback на корреляцию
                corr_with_target = X_numeric.corrwith(y).abs().sort_values(ascending=False)
                selected_features.extend(corr_with_target.head(max_features-5).index.tolist())
        
        # Добавляем важные текстовые признаки
        selected_features.extend(text_features[:5])
        
        # Ограничиваем до max_features
        selected_features = selected_features[:max_features]
        
        print(f"Выбрано {len(selected_features)} признаков")
        return selected_features
    
    def visualize_stage_results(self, model, X, y, features, stage_name):
        """Визуализация результатов стадии"""
        # Предсказания
        preds = model.predict(X)
        probas = model.predict_proba(X)[:, 1]
        
        # Создаем фигуру с подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Результаты модели {stage_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Feature Importance
        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
            feat_imp = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=True).tail(10)
            
            axes[0,1].barh(feat_imp['feature'], feat_imp['importance'])
            axes[0,1].set_title('Top 10 Feature Importance')
            axes[0,1].set_xlabel('Importance')
        
        # 3. Probability Distribution
        axes[1,0].hist(probas[y==0], bins=50, alpha=0.7, label='Class 0', color='red')
        axes[1,0].hist(probas[y==1], bins=50, alpha=0.7, label='Class 1', color='green')
        axes[1,0].axvline(self.hard_threshold_low, color='blue', linestyle='--', label=f'Low threshold ({self.hard_threshold_low})')
        axes[1,0].axvline(self.hard_threshold_high, color='blue', linestyle='--', label=f'High threshold ({self.hard_threshold_high})')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Probability Distribution')
        axes[1,0].legend()
        
        # 4. Classification Report (текст)
        cr = classification_report(y, preds, output_dict=True)
        report_text = f"""Classification Report:
        
Precision (Class 0): {cr['0']['precision']:.3f}
Recall (Class 0): {cr['0']['recall']:.3f}
F1-Score (Class 0): {cr['0']['f1-score']:.3f}

Precision (Class 1): {cr['1']['precision']:.3f}
Recall (Class 1): {cr['1']['recall']:.3f}
F1-Score (Class 1): {cr['1']['f1-score']:.3f}

Overall F1-Score: {cr['macro avg']['f1-score']:.3f}
Accuracy: {cr['accuracy']:.3f}"""
        
        axes[1,1].text(0.1, 0.5, report_text, fontsize=10, verticalalignment='center',
                      transform=axes[1,1].transAxes, family='monospace')
        axes[1,1].set_title('Performance Metrics')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def filter_confident_predictions(self, model, X, y, df_original, stage_name):
        """Фильтрация уверенных предсказаний с корректной индексацией"""
        print(f"Фильтрация confident predictions для {stage_name}...")
        
        probas = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        
        # Определяем confident предсказания
        low_confident = (probas < self.hard_threshold_low) & (preds == 0) & (y == 0)
        high_confident = (probas > self.hard_threshold_high) & (preds == 1) & (y == 1)
        confident_mask = low_confident | high_confident
        
        # Получаем ID из исходного датафрейма
        all_ids = df_original['id'].to_list()
        
        # Разделяем на confident и remaining
        confident_ids = [all_ids[i] for i in range(len(confident_mask)) if confident_mask[i]]
        remaining_ids = [all_ids[i] for i in range(len(confident_mask)) if not confident_mask[i]]
        
        df_confident = df_original.filter(pl.col('id').is_in(confident_ids))
        df_remaining = df_original.filter(pl.col('id').is_in(remaining_ids))
        
        print(f"Confident: {len(confident_ids)}, Remaining: {len(remaining_ids)}")
        
        return df_confident, df_remaining
    
    def train_stage(self, df: pl.DataFrame, stage_name: str, max_features: int):
        """Обучение одной стадии с визуализацией"""
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ СТАДИИ {stage_name.upper()}")
        print(f"{'='*60}")
        print(f"Размер выборки: {len(df)}")
        
        # Feature engineering
        df_processed = self.advanced_feature_engineering(df, stage_name)
        df_pandas = df_processed.to_pandas()
        
        # Feature selection
        features = self.improved_feature_selection(df_pandas, stage_name, max_features)
        
        if len(features) == 0:
            print(f"Нет признаков для {stage_name}!")
            return None, df_pandas, features
        
        X = df_pandas[features].copy()
        y = df_pandas[self.target].copy()
        
        # Обработка текстовых признаков
        for col in features:
            if col in self.text_columns:
                X[col] = X[col].fillna('missing').astype(str)
        
        # Определяем категориальные признаки
        cat_features_idx = []
        text_features_idx = []
        
        for i, col in enumerate(features):
            if col in self.text_columns:
                text_features_idx.append(i)
            elif X[col].dtype == 'object' or (X[col].nunique() <= 50 and X[col].dtype in ['int64', 'int32']):
                cat_features_idx.append(i)
        
        print(f"Всего признаков: {len(features)}")
        print(f"Категориальных: {len(cat_features_idx)}")
        print(f"Текстовых: {len(text_features_idx)}")
        
        # Создаем Pool
        train_pool = Pool(X, y, cat_features=cat_features_idx, text_features=text_features_idx)
        
        # Настройки модели для разных стадий
        stage_params = self.base_catboost_params.copy()
        if stage_name == 'hard':
            stage_params.update({'iterations': 400, 'depth': 6, 'learning_rate': 0.1})
        elif stage_name == 'soft':
            stage_params.update({'iterations': 600, 'depth': 8, 'learning_rate': 0.08})
        else:  # error
            stage_params.update({'iterations': 800, 'depth': 10, 'learning_rate': 0.05})
        
        # Обучение модели
        model = CatBoostClassifier(**stage_params)
        model.fit(train_pool, verbose=False)
        
        # Cross-validation score
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=255)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            temp_pool = Pool(X_train, y_train, cat_features=cat_features_idx, text_features=text_features_idx)
            temp_model = CatBoostClassifier(**stage_params)
            temp_model.fit(temp_pool, verbose=False)
            
            preds = temp_model.predict(X_val)
            f1 = f1_score(y_val, preds)
            cv_scores.append(f1)
        
        cv_f1 = np.mean(cv_scores)
        print(f"CV F1-score: {cv_f1:.4f} ± {np.std(cv_scores):.4f}")
        
        # Сохраняем модель и признаки
        self.models[stage_name] = model
        self.features[stage_name] = features
        
        # Визуализация результатов
        self.visualize_stage_results(model, X, y, features, stage_name)
        
        return model, df_pandas, features
    
    def visualize_pipeline_overview(self, stage_stats):
        """Общая визуализация пайплайна"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Обзор пайплайна Text-Aware Classification', fontsize=16, fontweight='bold')
        
        # 1. Размер выборки по стадиям
        stages = list(stage_stats.keys())
        sample_sizes = [stage_stats[stage]['sample_size'] for stage in stages]
        
        axes[0,0].bar(stages, sample_sizes, color=['#ff7f0e', '#2ca02c', '#d62728'])
        axes[0,0].set_title('Размер выборки по стадиям')
        axes[0,0].set_ylabel('Количество образцов')
        for i, v in enumerate(sample_sizes):
            axes[0,0].text(i, v + max(sample_sizes)*0.01, str(v), ha='center', va='bottom')
        
        # 2. F1-score по стадиям
        f1_scores = [stage_stats[stage]['f1_score'] for stage in stages if 'f1_score' in stage_stats[stage]]
        if f1_scores:
            axes[0,1].bar(stages[:len(f1_scores)], f1_scores, color=['#ff7f0e', '#2ca02c', '#d62728'])
            axes[0,1].set_title('F1-Score по стадиям')
            axes[0,1].set_ylabel('F1-Score')
            axes[0,1].set_ylim(0, 1)
            for i, v in enumerate(f1_scores):
                axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Количество признаков по стадиям
        feature_counts = [stage_stats[stage]['feature_count'] for stage in stages if 'feature_count' in stage_stats[stage]]
        if feature_counts:
            axes[1,0].bar(stages[:len(feature_counts)], feature_counts, color=['#ff7f0e', '#2ca02c', '#d62728'])
            axes[1,0].set_title('Количество признаков по стадиям')
            axes[1,0].set_ylabel('Количество признаков')
            for i, v in enumerate(feature_counts):
                axes[1,0].text(i, v + max(feature_counts)*0.01, str(v), ha='center', va='bottom')
        
        # 4. Общая статистика
        total_samples = sum(sample_sizes)
        processing_reduction = (sample_sizes[0] - sample_sizes[-1]) / sample_sizes[0] * 100 if len(sample_sizes) > 1 else 0
        
        stats_text = f"""Общая статистика пайплайна:
        
Всего образцов в начале: {sample_sizes[0]:,}
Образцов на последней стадии: {sample_sizes[-1]:,}
Сокращение объема обработки: {processing_reduction:.1f}%

Количество стадий: {len(stages)}
Использование GPU: {'Да' if self.gpu_available else 'Нет'}
Текстовых признаков: {len(self.text_columns)}

Стратегия: Поэтапная фильтрация 
confident predictions с уменьшением
сложности задачи на каждой стадии"""
        
        axes[1,1].text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
                      transform=axes[1,1].transAxes, family='monospace')
        axes[1,1].set_title('Статистика пайплайна')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def fit(self, train_df: pl.DataFrame):
        """Полное обучение пайплайна"""
        print("🚀 ЗАПУСК ОПТИМИЗИРОВАННОГО TEXT-AWARE PIPELINE")
        print("="*80)
        
        # Определяем текстовые колонки
        self.identify_text_columns(train_df)
        
        # Заполняем пропуски
        expressions = []
        for col, dtype in train_df.schema.items():
            if col not in self.col_drop + [self.target]:
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    expressions.append(pl.col(col).fill_null(0).alias(col))
                elif dtype == pl.String:
                    expressions.append(pl.col(col).fill_null("missing").alias(col))
        
        if expressions:
            train_df = train_df.with_columns(expressions)
        
        current_df = train_df
        stages = [('hard', 25), ('soft', 35), ('error', 50)]
        stage_stats = {}
        
        for stage_name, max_features in stages:
            stage_stats[stage_name] = {'sample_size': len(current_df)}
            
            if len(current_df) < 50:  # Минимальный размер для обучения
                print(f"Недостаточно данных для стадии {stage_name}: {len(current_df)}")
                break
            
            # Обучение стадии
            model, df_pandas, features = self.train_stage(current_df, stage_name, max_features)
            
            if model is None:
                break
            
            stage_stats[stage_name].update({
                'feature_count': len(features),
                'f1_score': np.mean([f1_score(df_pandas[self.target], model.predict(df_pandas[features])) for _ in [1]])
            })
            
            # Фильтрация confident predictions
            X = df_pandas[features].copy()
            y = df_pandas[self.target].copy()
            
            # Обработка текстовых признаков для фильтрации
            for col in features:
                if col in self.text_columns:
                    X[col] = X[col].fillna('missing').astype(str)
            
            confident, remaining = self.filter_confident_predictions(model, X, y, current_df, stage_name)
            
            current_df = remaining
            gc.collect()
            
            if len(current_df) == 0:
                print("Все образцы обработаны уверенными предсказаниями!")
                break
        
        # Общая визуализация
        self.visualize_pipeline_overview(stage_stats)
        
        print("\n" + "="*80)
        print("✅ ОБУЧЕНИЕ ПАЙПЛАЙНА ЗАВЕРШЕНО!")
        print("="*80)
        
        return self
    
    def predict(self, test_df: pl.DataFrame):
        """Предсказание с поэтапной обработкой"""
        print("🔮 НАЧАЛО ПРЕДСКАЗАНИЯ")
        print("="*50)
        
        # Определяем текстовые колонки
        self.identify_text_columns(test_df)
        
        # Заполняем пропуски
        expressions = []
        for col, dtype in test_df.schema.items():
            if col not in self.col_drop:
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    expressions.append(pl.col(col).fill_null(0).alias(col))
                elif dtype == pl.String:
                    expressions.append(pl.col(col).fill_null("missing").alias(col))
        
        if expressions:
            test_df = test_df.with_columns(expressions)
        
        current_df = test_df
        all_predictions = {}
        
        for stage_name in ['hard', 'soft', 'error']:
            if stage_name not in self.models or len(current_df) == 0:
                continue
                
            print(f"Предсказание для стадии {stage_name}: {len(current_df)} образцов")
            
            model = self.models[stage_name]
            features = self.features[stage_name]
            
            # Feature engineering
            df_processed = self.advanced_feature_engineering(current_df, stage_name)
            df_pandas = df_processed.to_pandas()
            
            # Проверяем доступность признаков
            available_features = [f for f in features if f in df_pandas.columns]
            if len(available_features) == 0:
                print(f"Нет доступных признаков для {stage_name}")
                continue
            
            X_test = df_pandas[available_features].copy()
            
            # Обработка текстовых признаков
            for col in available_features:
                if col in self.text_columns:
                    X_test[col] = X_test[col].fillna('missing').astype(str)
                elif col in self.label_encoders:
                    # Применяем сохраненный encoder
                    X_test[col] = self.label_encoders[col].transform(X_test[col].fillna('missing'))
            
            # Предсказания
            probas = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
            
            ids = current_df['id'].to_list()
            
            # Определяем confident predictions
            low_confident = (probas < self.hard_threshold_low) & (preds == 0)
            high_confident = (probas > self.hard_threshold_high) & (preds == 1)
            confident_mask = low_confident | high_confident
            
            # Сохраняем confident predictions
            for i, test_id in enumerate(ids):
                if confident_mask[i]:
                    all_predictions[test_id] = preds[i]
            
            # Оставляем только неуверенные для следующей стадии
            remaining_ids = [ids[i] for i in range(len(confident_mask)) if not confident_mask[i]]
            current_df = current_df.filter(pl.col('id').is_in(remaining_ids))
            
            confident_count = np.sum(confident_mask)
            print(f"Confident predictions: {confident_count}, Remaining: {len(remaining_ids)}")
        
        # Финальные предсказания
        test_ids = test_df['id'].to_list()
        final_predictions = [all_predictions.get(test_id, 0) for test_id in test_ids]
        
        # Статистика
        coverage = len(all_predictions) / len(test_ids) * 100
        unique_vals, counts = np.unique(final_predictions, return_counts=True)
        
        print(f"\nСтатистика предсказаний:")
        print(f"Coverage: {coverage:.1f}%")
        for val, count in zip(unique_vals, counts):
            percentage = count / len(final_predictions) * 100
            print(f"Class {val}: {count} ({percentage:.1f}%)")
        
        print("✅ ПРЕДСКАЗАНИЕ ЗАВЕРШЕНО!")
        return np.array(final_predictions)


# Функция для запуска полного процесса
def run_optimized_pipeline(train_path: str, test_path: str):
    """
    Запуск оптимизированного пайплайна с полной обработкой
    """
    try:
        print("🔥 ЗАГРУЗКА ОПТИМИЗИРОВАННОГО PIPELINE")
        
        # Создание и обучение пайплайна
        pipeline = OptimizedTextAwarePipeline()
        
        # Загрузка данных
        print("📊 Загрузка тренировочных данных...")
        train_df = pl.read_csv(train_path)
        print(f"Размер тренировочной выборки: {len(train_df)}")
        
        # Обучение
        pipeline.fit(train_df)
        
        # Загрузка тестовых данных
        print("🧪 Загрузка тестовых данных...")
        test_df = pl.read_csv(test_path)
        print(f"Размер тестовой выборки: {len(test_df)}")
        
        # Предсказание
        predictions = pipeline.predict(test_df)
        
        # Создание submission
        submission = pd.DataFrame({
            'id': test_df['id'].to_list(),
            'prediction': predictions
        })
        
        submission.to_csv('optimized_submission.csv', index=False)
        print("\n✅ Результаты сохранены в optimized_submission.csv")
        print("🎉 PIPELINE ВЫПОЛНЕН УСПЕШНО!")
        
        return pipeline, predictions
        
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Пример использования
    train_path = "path/to/train.csv"  # Укажите путь к тренировочным данным
    test_path = "path/to/test.csv"    # Укажите путь к тестовым данным
    
    pipeline, predictions = run_optimized_pipeline(train_path, test_path)