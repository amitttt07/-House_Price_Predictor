# House Price Prediction Model - Professional Implementation
# Complete ML Training Script for Uttarakhand Housing Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class HousePricePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_column = 'price'
        self.best_model_name = ""
        
    def load_and_prepare_data(self, data_path=None):
        """Load and prepare the dataset for training"""
        print("Creating realistic Uttarakhand housing dataset...")
        
        # Create synthetic dataset based on Uttarakhand real estate patterns
        np.random.seed(42)
        n_samples = 5000
        
        # Define realistic data distributions
        cities = ['Dehradun', 'Haridwar', 'Roorkee', 'Rishikesh', 'Nainital', 'Mussoorie', 'Almora', 'Pithoragarh']
        locations = {
            'Dehradun': ['Rajpur Road', 'Sahastradhara Road', 'Clement Town', 'GMS Road', 'Patel Nagar', 'Race Course'],
            'Haridwar': ['SIDCUL', 'Jwalapur', 'Kankhal', 'Shivalik Nagar', 'Roshnabad', 'Bhel'],
            'Roorkee': ['Civil Lines', 'Malviya Chowk', 'Gandhi Nagar', 'Solani Road', 'IIT Campus'],
            'Rishikesh': ['Tapovan', 'Laxman Jhula', 'Ram Jhula', 'Dehradun Road', 'Muni ki Reti'],
            'Nainital': ['Mall Road', 'Tallital', 'Mallital', 'Bhimtal Road', 'Haldwani Road'],
            'Mussoorie': ['Mall Road', 'Library Chowk', 'Landour', 'Happy Valley', 'Cloud End'],
            'Almora': ['Mall Road', 'Upper Mall Road', 'Lower Mall Road', 'Dharanaula'],
            'Pithoragarh': ['Upper Pithoragarh', 'Lower Pithoragarh', 'Siltham']
        }
        
        data = []
        for i in range(n_samples):
            city = np.random.choice(cities)
            location = np.random.choice(locations[city])
            bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.35, 0.35, 0.15, 0.05])
            
            # Area based on BHK with realistic distributions
            area_base = {1: 450, 2: 850, 3: 1200, 4: 1600, 5: 2200}
            area = int(np.random.normal(area_base[bhk], area_base[bhk] * 0.2))
            area = max(300, min(area, 4000))
            
            bathrooms = min(bhk, np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.4, 0.25, 0.1, 0.05]))
            balcony = np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.25, 0.05])
            age = np.random.exponential(5)
            age = min(age, 30)
            
            status = np.random.choice(['Ready to Move', 'Under Construction'], p=[0.7, 0.3])
            
            # Price calculation with realistic factors
            base_prices = {
                'Dehradun': 4500, 'Haridwar': 3200, 'Roorkee': 2800, 'Rishikesh': 3800,
                'Nainital': 5200, 'Mussoorie': 6500, 'Almora': 2500, 'Pithoragarh': 2200
            }
            
            location_multipliers = {
                'Rajpur Road': 1.4, 'Sahastradhara Road': 1.3, 'Mall Road': 1.5,
                'SIDCUL': 1.2, 'Tapovan': 1.3, 'IIT Campus': 1.15
            }
            
            base_price = base_prices[city]
            location_mult = location_multipliers.get(location, 1.0)
            
            # BHK premium (higher BHK = higher price per sq ft)
            bhk_mult = {1: 0.85, 2: 1.0, 3: 1.15, 4: 1.3, 5: 1.45}[bhk]
            
            # Status adjustment
            status_mult = 1.1 if status == 'Ready to Move' else 0.95
            
            # Age depreciation
            age_mult = max(0.7, 1 - (age * 0.02))
            
            # Calculate price per sq ft
            price_per_sqft = base_price * location_mult * bhk_mult * status_mult * age_mult
            
            # Add noise for realism
            noise_factor = np.random.normal(1.0, 0.1)
            price_per_sqft *= noise_factor
            
            total_price = price_per_sqft * area
            
            data.append({
                'city': city,
                'location': location,
                'bhk': bhk,
                'area': area,
                'bathrooms': bathrooms,
                'balcony': balcony,
                'age': age,
                'status': status,
                'price': total_price
            })
        
        self.df = pd.DataFrame(data)
        print(f"Dataset created with {len(self.df)} properties")
        return self.df
    
    def explore_data(self):
        """Comprehensive data exploration and analysis"""
        print("\n=== DATASET OVERVIEW ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        print("\n=== BASIC STATISTICS ===")
        print(self.df.describe().round(2))
        
        print("\n=== PRICE DISTRIBUTION BY CITY ===")
        city_stats = self.df.groupby('city')['price'].agg(['mean', 'median', 'std']).round(2)
        print(city_stats)
        
        print("\n=== PRICE BY BHK ===")
        bhk_stats = self.df.groupby('bhk')['price'].agg(['mean', 'median', 'count']).round(2)
        print(bhk_stats)
        
        return self.df.head(10)
    
    def handle_missing_data(self):
        """Handle missing data intelligently"""
        print("\n=== HANDLING MISSING DATA ===")
        
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() == 0:
            print("✅ No missing values found!")
            return self.df
        
        # Handle missing values based on column type
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            if missing_count > 0:
                print(f"Handling {missing_count} missing values in {column}")
                
                if column == 'price':
                    self.df = self.df.dropna(subset=[column])
                elif column in ['bhk', 'bathrooms', 'balcony']:
                    mode_value = self.df[column].mode()[0]
                    self.df[column].fillna(mode_value, inplace=True)
                elif column in ['area', 'age']:
                    self.df[column] = self.df.groupby('city')[column].transform(
                        lambda x: x.fillna(x.median())
                    )
                elif column in ['city', 'location', 'status']:
                    mode_value = self.df[column].mode()[0]
                    self.df[column].fillna(mode_value, inplace=True)
        
        print("✅ Missing data handled successfully!")
        return self.df
    
    def feature_engineering(self):
        """Advanced feature engineering"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Price per square foot
        self.df['price_per_sqft'] = self.df['price'] / self.df['area']
        
        # Room efficiency
        self.df['bathroom_ratio'] = self.df['bathrooms'] / self.df['bhk']
        
        # Property age groups
        self.df['age_group'] = pd.cut(self.df['age'], 
                                     bins=[0, 2, 5, 10, 20, 100], 
                                     labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old'])
        
        # Area categories
        self.df['area_category'] = pd.cut(self.df['area'], 
                                         bins=[0, 600, 1000, 1500, 2500, 10000],
                                         labels=['Compact', 'Medium', 'Large', 'Premium', 'Luxury'])
        
        # Location tier based on average prices
        location_avg_prices = self.df.groupby('location')['price_per_sqft'].mean()
        location_tiers = pd.qcut(location_avg_prices, 
                                q=3, 
                                labels=['Tier3', 'Tier2', 'Tier1'])
        location_tier_dict = location_tiers.to_dict()
        self.df['location_tier'] = self.df['location'].map(location_tier_dict)
        
        # Amenity score
        self.df['amenity_score'] = (
            (self.df['balcony'] > 0).astype(int) +
            (self.df['bathrooms'] >= self.df['bhk']).astype(int) +
            (self.df['status'] == 'Ready to Move').astype(int)
        )
        
        print("✅ Feature engineering completed!")
        return self.df
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("\n=== PREPARING FEATURES FOR ML ===")
        
        categorical_features = ['city', 'location', 'status', 'age_group', 'area_category', 'location_tier']
        numerical_features = ['bhk', 'area', 'bathrooms', 'balcony', 'age', 'bathroom_ratio', 'amenity_score']
        
        X = self.df[categorical_features + numerical_features].copy()
        y = self.df[self.target_column].copy()
        
        # One-hot encoding for categorical variables
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        print(f"✅ Final feature matrix shape: {X_encoded.shape}")
        print(f"✅ Number of features: {len(self.feature_names)}")
        
        return X_encoded, y
    
    def train_model(self, X, y):
        """Train multiple models and select the best one"""
        print("\n=== TRAINING MODELS ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0)
        }
        
        best_model = None
        best_score = -float('inf')
        model_scores = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for linear models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                self.use_scaling = True
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.use_scaling = False
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            print(f"  MAE: ₹{mae:,.0f}, RMSE: ₹{rmse:,.0f}, R2: {r2:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.best_model_name = name
        
        self.model = best_model
        self.X_test = X_test
        self.y_test = y_test
        self.model_scores = model_scores
        
        print(f"\n🏆 Best Model: {self.best_model_name} with R2 score: {best_score:.4f}")
        return model_scores
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        # Make predictions
        if self.best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            X_test_scaled = self.scaler.transform(self.X_test)
            y_pred = self.model.predict(X_test_scaled)
        else:
            y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        print(f"📊 Final Model Performance:")
        print(f"   MAE: ₹{mae:,.0f}")
        print(f"   RMSE: ₹{rmse:,.0f}")
        print(f"   R² Score: {r2:.4f} ({r2*100:.1f}%)")
        print(f"   MAPE: {mape:.2f}%")
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 Top 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
    
    def predict_price(self, city, location, bhk, area, bathrooms, balcony, age, status):
        """Predict price for a single property"""
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'city': [city], 'location': [location], 'bhk': [bhk], 'area': [area],
            'bathrooms': [bathrooms], 'balcony': [balcony], 'age': [age], 'status': [status]
        })
        
        # Feature engineering
        input_data['bathroom_ratio'] = input_data['bathrooms'] / input_data['bhk']
        input_data['age_group'] = pd.cut(input_data['age'], 
                                        bins=[0, 2, 5, 10, 20, 100], 
                                        labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old'])
        input_data['area_category'] = pd.cut(input_data['area'], 
                                            bins=[0, 600, 1000, 1500, 2500, 10000],
                                            labels=['Compact', 'Medium', 'Large', 'Premium', 'Luxury'])
        input_data['location_tier'] = 'Tier2'  # Default
        input_data['amenity_score'] = (
            (input_data['balcony'] > 0).astype(int) +
            (input_data['bathrooms'] >= input_data['bhk']).astype(int) +
            (input_data['status'] == 'Ready to Move').astype(int)
        )
        
        # Prepare features
        categorical_features = ['city', 'location', 'status', 'age_group', 'area_category', 'location_tier']
        numerical_features = ['bhk', 'area', 'bathrooms', 'balcony', 'age', 'bathroom_ratio', 'amenity_score']
        
        X_input = input_data[categorical_features + numerical_features].copy()
        X_input_encoded = pd.get_dummies(X_input, columns=categorical_features, drop_first=True)
        
        # Ensure same features as training data
        for feature in self.feature_names:
            if feature not in X_input_encoded.columns:
                X_input_encoded[feature] = 0
        
        X_input_encoded = X_input_encoded[self.feature_names]
        
        # Make prediction
        if self.best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            X_input_scaled = self.scaler.transform(X_input_encoded)
            prediction = self.model.predict(X_input_scaled)[0]
        else:
            prediction = self.model.predict(X_input_encoded)[0]
        
        return max(0, prediction)
    
    def save_model(self, filename='house_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name
        }
        joblib.dump(model_data, filename)
        print(f"💾 Model saved as {filename}")

def main():
    """Main execution function"""
    print("🏡 PROFESSIONAL HOUSE PRICE PREDICTION MODEL")
    print("=" * 60)
    
    # Initialize model
    model = HousePricePredictionModel()
    
    # Load and prepare data
    df = model.load_and_prepare_data()
    
    # Explore data
    model.explore_data()
    
    # Handle missing data
    df = model.handle_missing_data()
    
    # Feature engineering
    df = model.feature_engineering()
    
    # Prepare features
    X, y = model.prepare_features()
    
    # Train models
    scores = model.train_model(X, y)
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    # Save model
    model.save_model()
    
    # Example predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    
    test_cases = [
        {
            'city': 'Dehradun', 'location': 'Rajpur Road', 'bhk': 3, 'area': 1200,
            'bathrooms': 2, 'balcony': 2, 'age': 3, 'status': 'Ready to Move'
        },
        {
            'city': 'Nainital', 'location': 'Mall Road', 'bhk': 2, 'area': 800,
            'bathrooms': 2, 'balcony': 1, 'age': 1, 'status': 'Ready to Move'
        },
        {
            'city': 'Haridwar', 'location': 'SIDCUL', 'bhk': 4, 'area': 1600,
            'bathrooms': 3, 'balcony': 2, 'age': 5, 'status': 'Under Construction'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        predicted_price = model.predict_price(**case)
        price_per_sqft = predicted_price / case['area']
        
        print(f"\n🏠 Test Case {i}:")
        print(f"   📍 {case['city']}, {case['location']}")
        print(f"   🏡 {case['bhk']} BHK, {case['area']} sq ft")
        print(f"   💰 Predicted Price: ₹{predicted_price/100000:.1f}L (₹{price_per_sqft:,.0f}/sq ft)")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("🚀 Your model is ready for deployment!")
    print("=" * 60)

if __name__ == "__main__":
    main()