import requests
import json

# Test the API directly
url = "http://localhost:5000/predict"

print("🧪 Testing Location-Based Pricing...")
print("=" * 60)

# Test 1: Premium Location
test1 = {
    "city": "Dehradun",
    "location": "Rajpur Road",
    "bhk": 3,
    "area": 1200,
    "bathrooms": 2,
    "balcony": 2,
    "age": 3,
    "status": "Ready to Move"
}

print("\n📍 Test 1: PREMIUM LOCATION (Rajpur Road)")
print("-" * 60)
response1 = requests.post(url, json=test1)
result1 = response1.json()

if result1.get('success'):
    print(f"✅ Price: {result1['formatted_price']}")
    print(f"✅ Per sq ft: ₹{result1['price_per_sqft']:,}")
    print(f"✅ Total: ₹{result1['predicted_price']:,}")
else:
    print(f"❌ Error: {result1.get('error')}")

# Test 2: Local Location
test2 = {
    "city": "Dehradun",
    "location": "Nehru Colony",
    "bhk": 3,
    "area": 1200,
    "bathrooms": 2,
    "balcony": 2,
    "age": 3,
    "status": "Ready to Move"
}

print("\n📍 Test 2: LOCAL LOCATION (Nehru Colony)")
print("-" * 60)
response2 = requests.post(url, json=test2)
result2 = response2.json()

if result2.get('success'):
    print(f"✅ Price: {result2['formatted_price']}")
    print(f"✅ Per sq ft: ₹{result2['price_per_sqft']:,}")
    print(f"✅ Total: ₹{result2['predicted_price']:,}")
else:
    print(f"❌ Error: {result2.get('error')}")

# Compare
print("\n" + "=" * 60)
print("📊 COMPARISON")
print("=" * 60)

if result1.get('success') and result2.get('success'):
    price1 = result1['predicted_price']
    price2 = result2['predicted_price']
    
    diff = price1 - price2
    diff_percent = (diff / price2) * 100
    
    print(f"Premium (Rajpur Road):  ₹{price1:,}")
    print(f"Local (Nehru Colony):   ₹{price2:,}")
    print(f"Difference:             ₹{diff:,}")
    print(f"Percentage:             {diff_percent:.1f}% MORE")
    
    if diff_percent > 50:
        print("\n✅ WORKING CORRECTLY! Premium location is significantly more expensive!")
    elif diff_percent > 20:
        print("\n⚠️ PARTIAL! Some difference but should be more (~60%)")
    else:
        print("\n❌ NOT WORKING! Prices are too similar")
else:
    print("❌ Could not compare - API errors")

print("=" * 60)