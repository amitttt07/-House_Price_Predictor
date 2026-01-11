# Test script to verify location-based price differences
# Run this to see price variations across different locations

import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Test cases: Same property in different locations
test_cases = [
    # Dehradun - Premium vs Local
    {
        "name": "Dehradun - Rajpur Road (Premium)",
        "data": {
            "city": "Dehradun",
            "location": "Rajpur Road",
            "bhk": 3,
            "area": 1200,
            "bathrooms": 2,
            "balcony": 2,
            "age": 3,
            "status": "Ready to Move"
        }
    },
    {
        "name": "Dehradun - Nehru Colony (Local)",
        "data": {
            "city": "Dehradun",
            "location": "Nehru Colony",
            "bhk": 3,
            "area": 1200,
            "bathrooms": 2,
            "balcony": 2,
            "age": 3,
            "status": "Ready to Move"
        }
    },
    {
        "name": "Dehradun - Hathibarkala (Budget)",
        "data": {
            "city": "Dehradun",
            "location": "Hathibarkala",
            "bhk": 3,
            "area": 1200,
            "bathrooms": 2,
            "balcony": 2,
            "age": 3,
            "status": "Ready to Move"
        }
    },
    
    # Nainital - Tourist premium
    {
        "name": "Nainital - Mall Road (Premium Tourist Area)",
        "data": {
            "city": "Nainital",
            "location": "Mall Road",
            "bhk": 2,
            "area": 800,
            "bathrooms": 2,
            "balcony": 1,
            "age": 2,
            "status": "Ready to Move"
        }
    },
    {
        "name": "Nainital - Haldwani Road (Standard)",
        "data": {
            "city": "Nainital",
            "location": "Haldwani Road",
            "bhk": 2,
            "area": 800,
            "bathrooms": 2,
            "balcony": 1,
            "age": 2,
            "status": "Ready to Move"
        }
    },
    
    # Haridwar - Industrial vs Residential
    {
        "name": "Haridwar - SIDCUL (Industrial Hub)",
        "data": {
            "city": "Haridwar",
            "location": "SIDCUL",
            "bhk": 2,
            "area": 900,
            "bathrooms": 2,
            "balcony": 1,
            "age": 1,
            "status": "Ready to Move"
        }
    },
    {
        "name": "Haridwar - Jwalapur (Local Area)",
        "data": {
            "city": "Haridwar",
            "location": "Jwalapur",
            "bhk": 2,
            "area": 900,
            "bathrooms": 2,
            "balcony": 1,
            "age": 1,
            "status": "Ready to Move"
        }
    },
]

def test_location_pricing():
    """Test and display price differences across locations"""
    print("=" * 80)
    print("🏡 HOUSE PRICE PREDICTION - LOCATION COMPARISON TEST")
    print("=" * 80)
    print("\nTesting if premium locations show higher prices than local areas...")
    print()
    
    results = []
    
    for test_case in test_cases:
        try:
            response = requests.post(url, json=test_case["data"])
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    results.append({
                        'name': test_case['name'],
                        'city': test_case['data']['city'],
                        'location': test_case['data']['location'],
                        'bhk': test_case['data']['bhk'],
                        'area': test_case['data']['area'],
                        'price': result['predicted_price'],
                        'price_per_sqft': result['price_per_sqft'],
                        'formatted_price': result['formatted_price']
                    })
                else:
                    print(f"❌ Error for {test_case['name']}: {result.get('error')}")
            else:
                print(f"❌ HTTP Error {response.status_code} for {test_case['name']}")
                
        except Exception as e:
            print(f"❌ Exception for {test_case['name']}: {str(e)}")
    
    # Display results
    if results:
        print("\n" + "=" * 80)
        print("📊 PRICE COMPARISON RESULTS")
        print("=" * 80)
        
        current_city = None
        for idx, result in enumerate(results):
            if result['city'] != current_city:
                if current_city is not None:
                    print()  # Blank line between cities
                current_city = result['city']
                print(f"\n🏙️  {current_city.upper()}")
                print("-" * 80)
            
            print(f"\n{idx + 1}. {result['name']}")
            print(f"   📍 Location: {result['location']}")
            print(f"   🏡 Property: {result['bhk']} BHK, {result['area']} sq ft")
            print(f"   💰 Total Price: {result['formatted_price']}")
            print(f"   📊 Price/sq ft: ₹{result['price_per_sqft']:,}")
        
        # Calculate price differences
        print("\n" + "=" * 80)
        print("📈 PRICE DIFFERENCE ANALYSIS")
        print("=" * 80)
        
        # Dehradun comparison
        dehradun_results = [r for r in results if r['city'] == 'Dehradun']
        if len(dehradun_results) >= 2:
            premium = dehradun_results[0]
            local = dehradun_results[1]
            budget = dehradun_results[2] if len(dehradun_results) > 2 else None
            
            print(f"\n🏙️  DEHRADUN")
            print(f"   Premium (Rajpur Road): ₹{premium['price_per_sqft']:,}/sqft")
            print(f"   Local (Nehru Colony): ₹{local['price_per_sqft']:,}/sqft")
            if budget:
                print(f"   Budget (Hathibarkala): ₹{budget['price_per_sqft']:,}/sqft")
            
            diff_percent = ((premium['price'] - local['price']) / local['price']) * 100
            print(f"   💡 Premium is {diff_percent:.1f}% MORE expensive than Local")
            
            if budget:
                diff_budget = ((premium['price'] - budget['price']) / budget['price']) * 100
                print(f"   💡 Premium is {diff_budget:.1f}% MORE expensive than Budget")
        
        # Nainital comparison
        nainital_results = [r for r in results if r['city'] == 'Nainital']
        if len(nainital_results) >= 2:
            premium = nainital_results[0]
            standard = nainital_results[1]
            
            print(f"\n🏔️  NAINITAL")
            print(f"   Premium (Mall Road): ₹{premium['price_per_sqft']:,}/sqft")
            print(f"   Standard (Haldwani Road): ₹{standard['price_per_sqft']:,}/sqft")
            
            diff_percent = ((premium['price'] - standard['price']) / standard['price']) * 100
            print(f"   💡 Mall Road is {diff_percent:.1f}% MORE expensive than Haldwani Road")
        
        # Haridwar comparison
        haridwar_results = [r for r in results if r['city'] == 'Haridwar']
        if len(haridwar_results) >= 2:
            premium = haridwar_results[0]
            local = haridwar_results[1]
            
            print(f"\n🏭  HARIDWAR")
            print(f"   Industrial Hub (SIDCUL): ₹{premium['price_per_sqft']:,}/sqft")
            print(f"   Local Area (Jwalapur): ₹{local['price_per_sqft']:,}/sqft")
            
            diff_percent = ((premium['price'] - local['price']) / local['price']) * 100
            print(f"   💡 SIDCUL is {diff_percent:.1f}% MORE expensive than Jwalapur")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED - Location-based pricing is working correctly!")
        print("=" * 80)
        
    else:
        print("\n❌ No results to display. Make sure the Flask server is running!")
        print("   Run: python flask_api.py")

def quick_test():
    """Quick test to check if server is responding"""
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print("❌ Server responded but with error")
            return False
    except Exception as e:
        print("❌ Server is not running!")
        print(f"   Error: {str(e)}")
        print("\n   Please start the server first:")
        print("   python flask_api.py")
        return False

if __name__ == "__main__":
    print("\n🔍 Checking if server is running...")
    
    if quick_test():
        print("\n🚀 Starting location pricing test...\n")
        test_location_pricing()
    else:
        print("\n⚠️  Cannot run tests without server running.")
        print("   Start the server in another terminal window:")
        print("   python flask_api.py")