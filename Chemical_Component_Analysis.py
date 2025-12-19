import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import output_file
output_file("cosmetic_tsne.html")


# ============================================
# TASK 1: Load Dataset
# ============================================
# Update this path to your actual file location
df = pd.read_csv("cosmetics.csv")

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nSample rows:")
print(df.sample(5))
print("\nProduct type counts:")
print(df['Label'].value_counts())

# ============================================
# TASK 2: Filter Moisturizers for Dry Skin
# ============================================
moisturizers = df[df['Label'] == 'Moisturizer']
moisturizers_dry = moisturizers[moisturizers['Dry'] == 1].copy()
moisturizers_dry.reset_index(drop=True, inplace=True)

print(f"\nFiltered {len(moisturizers_dry)} moisturizers for dry skin")
print(moisturizers_dry.head())

# ============================================
# TASK 3: Tokenize Ingredients
# ============================================
corpus = []
ingredient_idx = {}
idx = 0

for ingredients in moisturizers_dry['Ingredients']:
    tokens = ingredients.lower().split(', ')
    corpus.append(tokens)
    
    for token in tokens:
        if token not in ingredient_idx:
            ingredient_idx[token] = idx
            idx += 1

print(f"\nTotal unique ingredients: {len(ingredient_idx)}")
print(f"Total products: {len(corpus)}")

# ============================================
# TASK 4: Initialize Document-Term Matrix
# ============================================
M = len(corpus)              # number of products
N = len(ingredient_idx)      # number of unique ingredients

A = np.zeros((M, N))
print(f"\nIngredient matrix shape: {A.shape}")

# ============================================
# TASK 5: One-Hot Encoder Function
# ============================================
def oh_encoder(tokens):
    """Convert ingredient tokens to one-hot encoded vector"""
    x = np.zeros(N)
    for token in tokens:
        if token in ingredient_idx:
            x[ingredient_idx[token]] = 1
    return x

# ============================================
# TASK 6: Fill Ingredient Matrix
# ============================================
for i, tokens in enumerate(corpus):
    A[i] = oh_encoder(tokens)

print("Ingredient matrix populated successfully!")
print(f"Non-zero entries: {np.count_nonzero(A)}")

# ============================================
# TASK 7: Dimensionality Reduction (t-SNE)
# ============================================
print("\nApplying t-SNE (this may take a moment)...")

tsne = TSNE(
    n_components=2,
    learning_rate=200,
    random_state=42,
    perplexity=min(30, len(moisturizers_dry) - 1)  # Adjust if dataset is small
)

tsne_features = tsne.fit_transform(A)

# Add coordinates to dataframe
moisturizers_dry['X'] = tsne_features[:, 0]
moisturizers_dry['Y'] = tsne_features[:, 1]

print("t-SNE transformation complete!")

# ============================================
# TASK 8: Create Bokeh Visualization
# ============================================
source = ColumnDataSource(moisturizers_dry)

p = figure(
    title="t-SNE Visualization of Moisturizers for Dry Skin",
    x_axis_label="t-SNE Dimension 1",
    y_axis_label="t-SNE Dimension 2",
    width=900,
    height=700,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

p.circle(
    x='X',
    y='Y',
    source=source,
    size=10,
    alpha=0.6,
    color='navy'
)

# ============================================
# TASK 9: Add Interactive Hover Tool
# ============================================
hover = HoverTool(tooltips=[
    ("Product", "@Name"),
    ("Brand", "@Brand"),
    ("Price", "$@Price"),
    ("Rank", "@Rank")
])

p.add_tools(hover)

# Style improvements
p.title.text_font_size = "16pt"
p.xaxis.axis_label_text_font_size = "12pt"
p.yaxis.axis_label_text_font_size = "12pt"

# ============================================
# TASK 10: Display Interactive Plot
# ============================================
print("\nDisplaying interactive plot...")
show(p)

# ============================================
# TASK 11: Compare Similar Products
# ============================================
print("\n" + "="*60)
print("PRODUCT COMPARISON: Similar Items Analysis")
print("="*60)

# Example comparison - modify product names based on your dataset
comparison = moisturizers_dry.nsmallest(2, 'Rank')  # Top 2 products by rank

for idx, row in comparison.iterrows():
    print(f"\nProduct {idx + 1}: {row['Name']}")
    print(f"Brand: {row['Brand']}")
    print(f"Price: ${row['Price']}")
    print(f"Rank: {row['Rank']}")
    print(f"Ingredients: {row['Ingredients'][:200]}...")  # First 200 chars

# ============================================
# BONUS: Find Similar Products Function
# ============================================
def find_similar_products(product_name, top_n=5):
    """Find top N similar products based on ingredient similarity"""
    try:
        # Get product index
        idx = moisturizers_dry[moisturizers_dry['Name'] == product_name].index[0]
        
        # Calculate cosine similarity with all products
        product_vector = A[idx]
        similarities = np.dot(A, product_vector) / (
            np.linalg.norm(A, axis=1) * np.linalg.norm(product_vector)
        )
        
        # Get top N similar (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        print(f"\nTop {top_n} products similar to '{product_name}':")
        print("-" * 60)
        
        for i, sim_idx in enumerate(similar_indices, 1):
            print(f"{i}. {moisturizers_dry.iloc[sim_idx]['Name']}")
            print(f"   Brand: {moisturizers_dry.iloc[sim_idx]['Brand']}")
            print(f"   Similarity: {similarities[sim_idx]:.3f}")
            print()
            
    except IndexError:
        print(f"Product '{product_name}' not found in dataset")

# Example usage (uncomment and modify with actual product name):
# find_similar_products("Your Product Name Here", top_n=5)

print("\n✅ All tasks completed successfully!")
print("📊 Interactive visualization is displayed above")
print("🔍 Use find_similar_products() function for recommendations")