import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore') 
# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class CustomerSegmentation:
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.optimal_clusters = None
    def load_data(self, file_path):
        """Load customer data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print("Data loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            print("\nFirst few rows:")
            print(self.data.head())
            print("\nDataset info:")
            print(self.data.info())
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    def create_sample_data(self, n_samples=200):
        """Create sample customer data for demonstration"""
        np.random.seed(42)
        # Generate synthetic customer data
        customer_ids = range(1, n_samples + 1)
        genders = np.random.choice(['Male', 'Female'], n_samples)
        ages = np.random.normal(40, 15, n_samples).astype(int)
        ages = np.clip(ages, 18, 80)  # Ensure realistic age range
        # Create income with some correlation to age
        annual_income = np.random.normal(50, 20, n_samples) + (ages - 40) * 0.5
        annual_income = np.clip(annual_income, 15, 120)  # Income in thousands
        # Create spending score with some inverse correlation to age
        spending_score = np.random.normal(50, 25, n_samples) + (50 - ages) * 0.3
        spending_score = np.clip(spending_score, 1, 100)  # Score 1-100
        self.data = pd.DataFrame({
            'CustomerID': customer_ids,
            'Gender': genders,
            'Age': ages,
            'Annual Income (k$)': annual_income.round(1),
            'Spending Score (1-100)': spending_score.round(0).astype(int)
        })
        print("Sample data created successfully!")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst few rows:")
        print(self.data.head())
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        # Basic statistics
        print("\nDescriptive Statistics:")
        print(self.data.describe())
        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        # Gender distribution
        if 'Gender' in self.data.columns:
            print("\nGender Distribution:")
            print(self.data['Gender'].value_counts())
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Data Exploratory Analysis', fontsize=16, fontweight='bold')
        # Age distribution
        axes[0, 0].hist(self.data['Age'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        # Annual Income distribution
        income_col = [col for col in self.data.columns if 'Income' in col][0]
        axes[0, 1].hist(self.data[income_col], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Annual Income Distribution')
        axes[0, 1].set_xlabel('Annual Income (k$)')
        axes[0, 1].set_ylabel('Frequency')
        # Spending Score distribution
        spending_col = [col for col in self.data.columns if 'Spending' in col or 'Score' in col][0]
        axes[0, 2].hist(self.data[spending_col], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('Spending Score Distribution')
        axes[0, 2].set_xlabel('Spending Score')
        axes[0, 2].set_ylabel('Frequency')
        # Scatter plot: Income vs Spending Score
        axes[1, 0].scatter(self.data[income_col], self.data[spending_col], alpha=0.6)
        axes[1, 0].set_title('Income vs Spending Score')
        axes[1, 0].set_xlabel('Annual Income (k$)')
        axes[1, 0].set_ylabel('Spending Score')
        # Scatter plot: Age vs Spending Score
        axes[1, 1].scatter(self.data['Age'], self.data[spending_col], alpha=0.6)
        axes[1, 1].set_title('Age vs Spending Score')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Spending Score')
        # Gender-based analysis (if available)
        if 'Gender' in self.data.columns:
            self.data.boxplot(column=spending_col, by='Gender', ax=axes[1, 2])
            axes[1, 2].set_title('Spending Score by Gender')
            axes[1, 2].set_xlabel('Gender')
            axes[1, 2].set_ylabel('Spending Score')
        else:
            axes[1, 2].axis('off')
        plt.tight_layout()
        plt.show()
    def find_optimal_clusters(self, max_clusters=10, features=None):
        """Find optimal number of clusters using Elbow Method and Silhouette Score"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        # Select features for clustering
        if features is None:
            # Use numerical columns (excluding CustomerID)
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if 'CustomerID' not in col and 'ID' not in col]
        print(f"\nUsing features for clustering: {features}")
        # Prepare data
        X = self.data[features].copy()
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(X)
        # Calculate WCSS for different number of clusters
        wcss = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        print("\nCalculating optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            wcss.append(kmeans.inertia_)
            # Calculate silhouette score
            sil_score = silhouette_score(self.scaled_data, kmeans.labels_)
            silhouette_scores.append(sil_score)
            print(f"K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette Score={sil_score:.3f}")
        # Find optimal k using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        self.optimal_clusters = optimal_k
        # Plot Elbow Method and Silhouette Score
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Elbow Method
        ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method For Optimal k', fontweight='bold')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Within Cluster Sum of Squares (WCSS)')
        ax1.grid(True, alpha=0.3)
        # Silhouette Score
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.axvline(x=optimal_k, color='green', linestyle='--', alpha=0.7, 
                   label=f'Optimal k = {optimal_k}')
        ax2.set_title('Silhouette Score For Different k', fontweight='bold')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"\nOptimal number of clusters based on Silhouette Score: {optimal_k}")
        return optimal_k
    def perform_clustering(self, n_clusters=None, features=None):
        """Perform K-means clustering"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        # Use optimal clusters if not specified
        if n_clusters is None:
            if self.optimal_clusters is None:
                self.find_optimal_clusters()
            n_clusters = self.optimal_clusters
        # Select features for clustering
        if features is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if 'CustomerID' not in col and 'ID' not in col]
        print(f"\nPerforming K-means clustering with {n_clusters} clusters...")
        print(f"Features used: {features}")
        # Prepare and scale data if not already done
        X = self.data[features].copy()
        if self.scaled_data is None:
            self.scaled_data = self.scaler.fit_transform(X)
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(self.scaled_data)
        # Add cluster labels to original data
        self.data['Cluster'] = cluster_labels
        # Calculate silhouette score
        sil_score = silhouette_score(self.scaled_data, cluster_labels)
        print(f"Clustering completed!")
        print(f"Silhouette Score: {sil_score:.3f}")
        return cluster_labels
    def analyze_clusters(self):
        """Analyze and interpret the clusters"""
        if 'Cluster' not in self.data.columns:
            print("No clustering performed yet. Please run perform_clustering() first.")
            return
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS")
        print("="*50)
        # Cluster sizes
        cluster_counts = self.data['Cluster'].value_counts().sort_index()
        print("\nCluster Sizes:")
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} customers ({count/len(self.data)*100:.1f}%)")
        # Cluster characteristics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col not in ['CustomerID', 'Cluster']]
        print("\nCluster Characteristics (Mean Values):")
        cluster_summary = self.data.groupby('Cluster')[features].mean()
        print(cluster_summary.round(2))
        # Create visualizations
        n_features = len(features)
        n_clusters = len(cluster_counts)
        # Scatter plots for key relationships
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Clusters Visualization', fontsize=16, fontweight='bold')
        # Get income and spending columns
        income_col = [col for col in features if 'Income' in col][0]
        spending_col = [col for col in features if 'Spending' in col or 'Score' in col][0]
        # Income vs Spending Score by Cluster
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        for i, cluster in enumerate(sorted(self.data['Cluster'].unique())):
            cluster_data = self.data[self.data['Cluster'] == cluster]
            axes[0, 0].scatter(cluster_data[income_col], cluster_data[spending_col], 
                             c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=50)
        axes[0, 0].set_title('Income vs Spending Score by Cluster')
        axes[0, 0].set_xlabel('Annual Income (k$)')
        axes[0, 0].set_ylabel('Spending Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        # Age vs Spending Score by Cluster
        for i, cluster in enumerate(sorted(self.data['Cluster'].unique())):
            cluster_data = self.data[self.data['Cluster'] == cluster]
            axes[0, 1].scatter(cluster_data['Age'], cluster_data[spending_col], 
                             c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=50)
        axes[0, 1].set_title('Age vs Spending Score by Cluster')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Spending Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        # Age vs Income by Cluster
        for i, cluster in enumerate(sorted(self.data['Cluster'].unique())):
            cluster_data = self.data[self.data['Cluster'] == cluster]
            axes[1, 0].scatter(cluster_data['Age'], cluster_data[income_col], 
                             c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=50)
        axes[1, 0].set_title('Age vs Income by Cluster')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Annual Income (k$)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        # Cluster distribution
        axes[1, 1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                      autopct='%1.1f%%', colors=colors[:len(cluster_counts)])
        axes[1, 1].set_title('Customer Distribution by Cluster')
        plt.tight_layout()
        plt.show()
        # Box plots for each feature by cluster
        fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 6))
        if len(features) == 1:
            axes = [axes]
        for i, feature in enumerate(features):
            self.data.boxplot(column=feature, by='Cluster', ax=axes[i])
            axes[i].set_title(f'{feature} by Cluster')
            axes[i].set_xlabel('Cluster')
        plt.suptitle('Feature Distribution by Cluster', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    def interpret_clusters(self):
        """Provide business interpretation of clusters"""
        if 'Cluster' not in self.data.columns:
            print("No clustering performed yet. Please run perform_clustering() first.")
            return
        print("\n" + "="*50)
        print("BUSINESS INTERPRETATION")
        print("="*50)
        # Get cluster characteristics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col not in ['CustomerID', 'Cluster']]
        cluster_summary = self.data.groupby('Cluster')[features].mean()
        # Interpret each cluster
        for cluster in sorted(self.data['Cluster'].unique()):
            cluster_data = cluster_summary.loc[cluster]
            size = len(self.data[self.data['Cluster'] == cluster])
            print(f"\n🏷️  CLUSTER {cluster} ({size} customers, {size/len(self.data)*100:.1f}%)")
            print("-" * 40)
            avg_age = cluster_data['Age']
            avg_income = cluster_data[[col for col in features if 'Income' in col][0]]
            avg_spending = cluster_data[[col for col in features if 'Spending' in col or 'Score' in col][0]]
            print(f"Average Age: {avg_age:.1f} years")
            print(f"Average Income: ${avg_income:.1f}k")
            print(f"Average Spending Score: {avg_spending:.1f}/100")
            # Provide interpretation based on characteristics
            if avg_income >= 70 and avg_spending >= 70:
                interpretation = "🌟 High Value Customers - High income, high spending. Target for premium products and VIP services."
            elif avg_income >= 70 and avg_spending <= 40:
                interpretation = "💰 Careful Spenders - High income, low spending. Need targeted promotions to increase engagement."
            elif avg_income <= 40 and avg_spending >= 70:
                interpretation = "🛍️ Enthusiastic Shoppers - Low income, high spending. Price-sensitive but loyal customers."
            elif avg_income <= 40 and avg_spending <= 40:
                interpretation = "💡 Budget Conscious - Low income, low spending. Focus on value deals and basic products."
            else:
                interpretation = "⚖️ Average Customers - Moderate income and spending. Standard marketing approaches work well."
            print(f"Interpretation: {interpretation}")
    def save_results(self, filename='customer_segments.csv'):
        """Save clustering results to CSV"""
        if 'Cluster' not in self.data.columns:
            print("No clustering performed yet. Please run perform_clustering() first.")
            return
        self.data.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
    def predict_new_customer(self, customer_data):
        """Predict cluster for new customer data"""
        if self.kmeans is None:
            print("No model trained yet. Please run perform_clustering() first.")
            return None
        # Prepare new customer data
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col not in ['CustomerID', 'Cluster']]
        new_customer_scaled = self.scaler.transform([customer_data[features]])
        cluster = self.kmeans.predict(new_customer_scaled)[0]
        print(f"New customer predicted to belong to Cluster {cluster}")
        return cluster
# Example usage
def main():
    # Initialize the customer segmentation class
    cs = CustomerSegmentation()
    # Option 1: Load data from file (uncomment if you have the dataset)
    # success = cs.load_data('Mall_Customers.csv')
    # if not success:
    #     print("Using sample data instead...")
    #     cs.create_sample_data()
    # Option 2: Create sample data for demonstration
    cs.create_sample_data(n_samples=300)
    # Perform exploratory data analysis
    cs.exploratory_analysis()
    # Find optimal number of clusters
    optimal_k = cs.find_optimal_clusters(max_clusters=8)
    # Perform clustering
    cs.perform_clustering(n_clusters=optimal_k)
    # Analyze clusters
    cs.analyze_clusters()
    # Interpret clusters for business use
    cs.interpret_clusters()
    # Save results
    cs.save_results('customer_segments_results.csv')
    # Example: Predict cluster for a new customer
    new_customer = {
        'Age': 35,
        'Annual Income (k$)': 60.0,
        'Spending Score (1-100)': 75
    }
    cs.predict_new_customer(new_customer)
if __name__ == "__main__":
    main()