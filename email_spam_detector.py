import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EmailSpamDetector:
    """
    A machine learning model to detect spam emails using text classification
    """
    
    def __init__(self, filepath):
        """Initialize the spam detector with a dataset"""
        self.df = pd.read_csv(filepath)
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', lowercase=True)
        self.X_train_vec = None
        self.X_test_vec = None
        
    def display_dataset_info(self):
        """Display information about the dataset"""
        print("=" * 70)
        print("EMAIL SPAM DETECTION - DATASET OVERVIEW")
        print("=" * 70)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumn Names:\n{self.df.columns.tolist()}")
        print(f"\nFirst Few Rows:")
        print(self.df.head(3))
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        
        # Spam distribution
        if 'Label' in self.df.columns:
            spam_dist = self.df['Label'].value_counts()
            print(f"\nSpam Distribution:")
            print(spam_dist)
            print(f"\nSpam Percentage: {(spam_dist.get(1, 0) / len(self.df)) * 100:.2f}%")
            print(f"Ham Percentage: {(spam_dist.get(0, 0) / len(self.df)) * 100:.2f}%")
    
    def analyze_text_features(self):
        """Analyze text features in emails"""
        print("\n" + "=" * 70)
        print("TEXT FEATURE ANALYSIS")
        print("=" * 70)
        
        if 'Email' in self.df.columns and 'Label' in self.df.columns:
            self.df['Email_Length'] = self.df['Email'].apply(len)
            self.df['Word_Count'] = self.df['Email'].apply(lambda x: len(x.split()))
            self.df['Special_Chars'] = self.df['Email'].apply(lambda x: sum(1 for c in x if not c.isalnum()))
            
            print("\nEmail Length Statistics:")
            print(f"  Average: {self.df['Email_Length'].mean():.2f} characters")
            print(f"  Min: {self.df['Email_Length'].min()} characters")
            print(f"  Max: {self.df['Email_Length'].max()} characters")
            
            print("\nWord Count Statistics:")
            print(f"  Average: {self.df['Word_Count'].mean():.2f} words")
            print(f"  Min: {self.df['Word_Count'].min()} words")
            print(f"  Max: {self.df['Word_Count'].max()} words")
            
            print("\nSpam vs Ham Comparison:")
            comparison = self.df.groupby('Label')[['Email_Length', 'Word_Count', 'Special_Chars']].mean()
            comparison.index = ['Ham', 'Spam']
            print(comparison)
            
            # Find common spam words
            spam_emails = self.df[self.df['Label'] == 1]['Email']
            ham_emails = self.df[self.df['Label'] == 0]['Email']
            
            print("\nCommon Spam Indicators:")
            spam_indicators = ['free', 'click', 'urgent', 'winner', 'congratulations', 
                             'limited', 'offer', 'now', 'verify', 'confirm']
            for word in spam_indicators:
                spam_count = sum(1 for email in spam_emails if word.lower() in email.lower())
                spam_pct = (spam_count / len(spam_emails)) * 100 if len(spam_emails) > 0 else 0
                print(f"  '{word}': {spam_pct:.1f}% of spam emails")
    
    def preprocess_data(self):
        """Preprocess data for model training"""
        print("\n" + "=" * 70)
        print("DATA PREPROCESSING")
        print("=" * 70)
        
        if 'Email' not in self.df.columns or 'Label' not in self.df.columns:
            raise ValueError("'Email' and 'Label' columns required")
        
        # Prepare data
        X = self.df['Email']
        y = self.df['Label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ Data split: 80% training ({len(X_train)} emails), 20% testing ({len(X_test)} emails)")
        
        # Vectorize text
        self.X_train_vec = self.vectorizer.fit_transform(X_train)
        self.X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"✓ Text vectorized using TF-IDF")
        print(f"  Vocabulary size: {len(self.vectorizer.get_feature_names_out())} features")
        
        return self.X_train_vec, self.X_test_vec, y_train, y_test
    
    def train_naive_bayes(self, X_train_vec, X_test_vec, y_train, y_test):
        """Train Naive Bayes model"""
        print("\n" + "-" * 70)
        print("Training Naive Bayes Model...")
        print("-" * 70)
        
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
        
        # Evaluation
        self._evaluate_model(model, y_test, y_pred, y_pred_proba, "Naive Bayes")
        
        self.models['Naive Bayes'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return model
    
    def train_logistic_regression(self, X_train_vec, X_test_vec, y_train, y_test):
        """Train Logistic Regression model"""
        print("\n" + "-" * 70)
        print("Training Logistic Regression Model...")
        print("-" * 70)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
        
        # Evaluation
        self._evaluate_model(model, y_test, y_pred, y_pred_proba, "Logistic Regression")
        
        self.models['Logistic Regression'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return model
    
    def train_svm(self, X_train_vec, X_test_vec, y_train, y_test):
        """Train SVM model"""
        print("\n" + "-" * 70)
        print("Training Support Vector Machine (SVM) Model...")
        print("-" * 70)
        
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
        
        # Evaluation
        self._evaluate_model(model, y_test, y_pred, y_pred_proba, "SVM")
        
        self.models['SVM'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return model
    
    def train_random_forest(self, X_train_vec, X_test_vec, y_train, y_test):
        """Train Random Forest model"""
        print("\n" + "-" * 70)
        print("Training Random Forest Model...")
        print("-" * 70)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_vec, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
        
        # Evaluation
        self._evaluate_model(model, y_test, y_pred, y_pred_proba, "Random Forest")
        
        self.models['Random Forest'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return model
    
    def _evaluate_model(self, model, y_test, y_pred, y_pred_proba, model_name):
        """Helper function to evaluate model"""
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✓ Model trained successfully!")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} (False positive rate)")
        print(f"  Recall:    {recall:.4f} (Detection rate)")
        print(f"  F1-Score:  {f1:.4f} (Harmonic mean)")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        comparison_df = pd.DataFrame({
            'Model': self.models.keys(),
            'Accuracy': [m['accuracy'] for m in self.models.values()],
            'Precision': [m['precision'] for m in self.models.values()],
            'Recall': [m['recall'] for m in self.models.values()],
            'F1-Score': [m['f1'] for m in self.models.values()],
            'ROC-AUC': [m['roc_auc'] for m in self.models.values()]
        })
        
        print(f"\n{comparison_df.to_string(index=False)}")
        
        # Find best model by F1-score
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.loc[best_idx]
        print(f"\n✓ Best Model: {best_model['Model']} (F1 = {best_model['F1-Score']:.4f})")
        
        return comparison_df
    
    def plot_confusion_matrices(self, y_test, save_path='confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Confusion Matrices - Spam Detection', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (model_name, model_data) in enumerate(self.models.items()):
            ax = axes[idx]
            predictions = model_data['predictions']
            cm = confusion_matrix(y_test, predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            
            ax.set_title(f'{model_name}\n(Accuracy: {model_data["accuracy"]:.4f})', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved as '{save_path}'")
        plt.show()
    
    def plot_roc_curves(self, y_test, save_path='roc_curves.png'):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
        
        for (model_name, model_data), color in zip(self.models.items(), colors):
            y_pred_proba = model_data['probabilities']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = model_data['roc_auc']
            
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Spam Detection Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved as '{save_path}'")
        plt.show()
    
    def plot_metrics_comparison(self, save_path='metrics_comparison.png'):
        """Plot metrics comparison for all models"""
        comparison_df = pd.DataFrame({
            'Model': self.models.keys(),
            'Accuracy': [m['accuracy'] for m in self.models.values()],
            'Precision': [m['precision'] for m in self.models.values()],
            'Recall': [m['recall'] for m in self.models.values()],
            'F1-Score': [m['f1'] for m in self.models.values()]
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        ax.bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy', color='#3498DB')
        ax.bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision', color='#E74C3C')
        ax.bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall', color='#2ECC71')
        ax.bar(x + 1.5*width, comparison_df['F1-Score'], width, label='F1-Score', color='#F39C12')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved as '{save_path}'")
        plt.show()
    
    def predict_email(self, email_text):
        """Predict if an email is spam"""
        print("\n" + "=" * 70)
        print("EMAIL SPAM PREDICTION")
        print("=" * 70)
        
        email_vec = self.vectorizer.transform([email_text])
        
        print(f"\nEmail Preview: {email_text[:100]}...")
        print(f"\nPredictions from different models:")
        
        predictions = {}
        for model_name, model_data in self.models.items():
            model = model_data['model']
            pred = model.predict(email_vec)[0]
            prob = model.predict_proba(email_vec)[0][1]
            
            label = "SPAM 🚫" if pred == 1 else "HAM ✅"
            predictions[model_name] = (label, prob)
            
            print(f"  {model_name:20s}: {label} (Confidence: {prob*100:.2f}%)")
        
        # Ensemble prediction
        avg_spam_prob = np.mean([p[1] for p in predictions.values()])
        ensemble_label = "SPAM 🚫" if avg_spam_prob > 0.5 else "HAM ✅"
        
        print(f"\n✓ Ensemble Prediction: {ensemble_label}")
        print(f"  Average Spam Probability: {avg_spam_prob*100:.2f}%")
        
        return predictions, ensemble_label
    
    def generate_report(self, X_train_vec, X_test_vec, y_train, y_test):
        """Generate complete analysis report"""
        print("\n" + "=" * 70)
        print("EMAIL SPAM DETECTION - COMPLETE REPORT")
        print("=" * 70)
        
        self.display_dataset_info()
        self.analyze_text_features()
        
        print("\n" + "=" * 70)
        print("MODEL TRAINING")
        print("=" * 70)
        
        self.train_naive_bayes(X_train_vec, X_test_vec, y_train, y_test)
        self.train_logistic_regression(X_train_vec, X_test_vec, y_train, y_test)
        self.train_svm(X_train_vec, X_test_vec, y_train, y_test)
        self.train_random_forest(X_train_vec, X_test_vec, y_train, y_test)
        
        self.compare_models()
        
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS...")
        print("=" * 70)
        
        self.plot_confusion_matrices(y_test)
        self.plot_roc_curves(y_test)
        self.plot_metrics_comparison()
        
        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE!")
        print("=" * 70)


if __name__ == "__main__":
    # Initialize the spam detector
    detector = EmailSpamDetector('spam_emails.csv')
    
    # Preprocess data
    X_train_vec, X_test_vec, y_train, y_test = detector.preprocess_data()
    
    # Generate complete report
    detector.generate_report(X_train_vec, X_test_vec, y_train, y_test)
    
    # Example: Predict spam for a new email
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    spam_example = "Congratulations! You won $1,000,000! Click here now to claim your prize!"
    ham_example = "Hi John, Let's schedule a meeting for next Tuesday. Are you available at 2 PM?"
    
    print("\n\nExample 1: Likely Spam")
    detector.predict_email(spam_example)
    
    print("\n\nExample 2: Likely Ham")
    detector.predict_email(ham_example)
