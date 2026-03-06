#%% md
# # Task 1: Data Inspection and Preparation
#%% md
# ## a) Understanding and Pruning the Data
#%% md
# loading the data
#%%
import pandas as pd
display = print # display is notebook only

orders = pd.read_csv('orders.csv')
order_products = pd.read_csv('order_products.csv')
aisles = pd.read_csv('aisles.csv')
products = pd.read_csv('products.csv')
departments = pd.read_csv('departments.csv')
#%% md
# checking heads
#%%
display(orders.head())
display(order_products.head())
display(aisles.head())
display(products.head())
display(departments.head())
#%% md
# sampling users
#%%
SEED = 333
SAMPLE_SIZE = 10000

# original item counts
print("BEFORE FILTERING")
print(f"len(orders): {len(orders)}")
print(f"len(order_products): {len(order_products)}")

# random sampling
sample_user_id = orders['user_id'].drop_duplicates().sample(n=SAMPLE_SIZE, random_state=SEED)

# filter using the sampling
orders_reduced = orders[orders['user_id'].isin(sample_user_id)]
order_products_reduced = order_products[order_products['order_id'].isin(orders_reduced['order_id'])]

# new item counts
print("AFTER FILTERING")
print(f"len(orders): {len(orders_reduced)}")
print(f"len(order_products): {len(order_products_reduced)}")
#%% md
# ## b) Constructing Transactions
#%% md
# constructing transactions
#%%
# merge with products (for product names)
df_merged = order_products_reduced.merge(products[['product_id', 'product_name']], on='product_id')

# group by order number
transactions = df_merged.groupby('order_id')['product_name'].apply(list).values.tolist()

# check head
print("example transaction:")
print(transactions[SEED])
#%% md
# # Task 2: Mining Association Rules
#%% md
# ## a) Exploring the Dataset
#%% md
# generate ruleset
#%%
from apyori import apriori

MIN_SUPPORT = 0.001
MIN_CONFIDENCE = 0.2

rules = apriori(transactions, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, min_lift=3)
results = list(rules)

print(f"len(results): {len(results)}")
#%% md
# convert ruleset to dataframe
#%%
def extract_rules(rules_as_list):
    rules_data = []
    for result in rules_as_list:
        for ordered_stat in result.ordered_statistics:
            rules_data.append({
                'Base': list(ordered_stat.items_base),
                'Add': list(ordered_stat.items_add),
                'Support': result.support,
                'Confidence': ordered_stat.confidence,
                'Lift': ordered_stat.lift
            })
    return pd.DataFrame(rules_data)

rules_df = extract_rules(results)
display(rules_df.sort_values(by='Lift', ascending=False).head(10))
#%% md
# ## b) Identifying Market Insights
#%% md
# product rule with the highest confidence & support
#%%
high_confidence = rules_df[rules_df['Confidence'] > 0.4].sort_values(by='Confidence', ascending=False)
high_support = rules_df.sort_values(by='Support', ascending=False)

print("highest confidence rule:")
display(high_confidence.head(1))
print("highest support rule:")
display(high_support.head(1))
#%% md
# top sunday morning rule
#%%
# filter for sunday morning
filtered_ids = orders_reduced[
    (orders_reduced['order_dow'] == 0) &
    (orders_reduced['order_hour_of_day'].between(6, 10))
]['order_id']

# transaction list
filtered_df = df_merged[df_merged['order_id'].isin(filtered_ids)]
filtered_transactions = filtered_df.groupby('order_id')['product_name'].apply(list).values.tolist()

# generate ruleset
MIN_SUPPORT = 0.004
MIN_CONFIDENCE = 0.2
filtered_rules_results = list(apriori(filtered_transactions, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE))

# convert to df and display
filtered_rules_df = extract_rules(filtered_rules_results)

print(f"len(filtered_rules_df): {len(filtered_rules_df)}")
print("top sunday morning rule:")
display(filtered_rules_df.sort_values(by='Confidence', ascending=False).head(1))
#%% md
# department rule
#%%
# merge with departments (for department names)
df_dept = order_products_reduced.merge(products[['product_id', 'department_id']], on='product_id')
df_dept = df_dept.merge(departments, on='department_id')

# aggregate
grouped = df_dept.groupby('order_id')['department']

# construct transactions (and remove duplicates)
dept_transactions = []
for name, group in grouped:
    unique_departments = list(set(group))
    dept_transactions.append(unique_departments)

# generate ruleset
MIN_SUPPORT = 0.1
MIN_CONFIDENCE = 0.5
dept_results = list(apriori(dept_transactions, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE))

# extract and display ruleset
dept_rules_df = extract_rules(dept_results)

# find top confidence rule
top_rule = dept_rules_df.sort_values(by='Confidence', ascending=False).head(1)
print("top departments rule:")
display(top_rule)