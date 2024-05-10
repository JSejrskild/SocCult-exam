import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('Fertility_clean.csv')  # Replace with the actual filename and path

# STEP 01
# Define Agent class
class Agent:
    def __init__(self, municipality, fertility_rate, socio_economic_class, age, children):
        self.municipality = municipality
        self.fertility_rate = fertility_rate
        self.socio_economic_class = socio_economic_class
        self.age = age
        self.children = 0

# STEP 02
# Define a function to create agents for a municipality
def create_agents_for_municipality(municipality_class, num_agents):
    agents = []
    for agent_data in municipality_class.agents_data:
        fertility_rate = agent_data['fertility_rate']
        socio_economic_class = agent_data['socio_economic_class']
        age = agent_data['age']
        agents.append(Agent(
            municipality=municipality_class,
            fertility_rate=fertility_rate,
            socio_economic_class=socio_economic_class,
            age=age,
            children=0
        ))
    return agents[:num_agents]  # Adjust to create agents based on percentage

# STEP 03
# Define a function to create Municipality classes dynamically
def create_municipality_class(name, fertility_rate, initial_population_percentage, socio_economic_class, socio_economic_weights):
    municipality_class = type(name, (object,), {
        'name': name,
        'fertility_rate': fertility_rate,
        'initial_population_percentage': initial_population_percentage,
        'socio_economic_class': socio_economic_class,
        'agents_data': []  # List to store agent data within the municipality
    })

# STEP 04
    # Generate agent data within the municipality
    mean_fertility_rate = fertility_rate
    std_dev_fertility_rate = 0.1 * fertility_rate  # Adjust the standard deviation as needed
    total_agents = 10000  # Total number of agents per municipality
    num_agents = int(total_agents * (initial_population_percentage / 100))

    for _ in range(num_agents):
        # Sample fertility rate from a normal distribution around the mean fertility rate
        fertility_rate_sampled = np.random.normal(mean_fertility_rate, std_dev_fertility_rate)
        # Ensure fertility rate is non-negative
        fertility_rate_sampled = max(0, fertility_rate_sampled)

        # Sample socio-economic class weighted around the initial socio-economic class
        socio_economic_class_sampled = np.random.choice(['1', '2', '3'], p=socio_economic_weights)

        # Append agent data to the list
        municipality_class.agents_data.append({
            'fertility_rate': fertility_rate_sampled,
            'socio_economic_class': socio_economic_class_sampled,
            'age': 15  # Assuming a constant initial age for agents
        })

    # Create agents for the municipality
    municipality_class.agents = create_agents_for_municipality(municipality_class, num_agents)
    
    return municipality_class

# STEP 05
# Create Municipality classes dynamically based on the dataset
municipality_classes = {}
for index, row in data.iterrows():
    municipality_name = row['municipalities']
    fertility_rate = row['fertilitetsrate']
    initial_population_percentage = row['Percent'] 
    socio_economic_class = row['econ_score']  # Assuming 'Socio-economic Class' is the column name
    
    # Define weights for socio-economic classes for this municipality
    if socio_economic_class == '1':
        socio_economic_weights = [0.6, 0.2, 0.2]  # Low socio-economic class has a weight of 0.6
    elif socio_economic_class == '2':
        socio_economic_weights = [0.2, 0.6, 0.2]  # Middle socio-economic class has a weight of 0.6
    else:
        socio_economic_weights = [0.2, 0.2, 0.6]  # High socio-economic class has a weight of 0.6
        
    municipality_class = create_municipality_class(municipality_name, fertility_rate, initial_population_percentage, socio_economic_class, socio_economic_weights)
    municipality_classes[municipality_name] = municipality_class


# STEP 06 - Create DF
# Initialize an empty list to collect agent attributes
agent_data_list = []

# Iterate through municipality classes and their agents
for municipality_name, municipality_class in municipality_classes.items():
    for agent in municipality_class.agents:
        # Collect agent attributes into a dictionary
        agent_data = {
            'Municipality': agent.municipality.name,
            'Fertility Rate': agent.fertility_rate,
            'socio_economic_class': agent.socio_economic_class,
            'Age': agent.age,
            'Children': agent.children
        }
        # Append agent data to the list
        agent_data_list.append(agent_data)

# Create DataFrame from the list of agent data
agents_df = pd.DataFrame(agent_data_list)

# Create age bins
age_bins = [(15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49)]

# Function to update age bins based on agent's age
def update_age_bins(agent_age):
    for age_bin in age_bins:
        if age_bin[0] <= agent_age <= age_bin[1]:
            return f"{age_bin[0]}-{age_bin[1]}"

# Update age bins for each agent
agents_df['Age Group'] = agents_df['Age'].apply(update_age_bins)

# Initialize columns for each age bin
for age_bin in age_bins:
    age_group = f"{age_bin[0]}-{age_bin[1]}"
    agents_df[age_group] = 0

# Function to update children count for each agent in age bins and total children
def update_children_count(row):
    age = row['Age']
    children = row['Children'] = row[[f"{age_bin[0]}-{age_bin[1]}" for age_bin in age_bins]].sum()  # Update the 'Children' column with the total children count
    age_group = update_age_bins(age)
    row[age_group] += children
    return row


# Apply the function to update children count for each agent
agents_df = agents_df.apply(update_children_count, axis=1)

# Reorder columns
agents_df = agents_df[['Municipality', 'Fertility Rate', 'socio_economic_class', 'Age', 'Children'] + [f"{age_bin[0]}-{age_bin[1]}" for age_bin in age_bins]]


def simulate_child_birth(agents_df, year):
    # Define probability weights for having a child based on age bins and socio-economic classes
    probability_weights = {
        '1': {'15-19': 0.003, '20-24': 0.03, '25-29': 0.082, '30-34': 0.059, '35-39': 0.021, '40-44': 0.004, '45-49': 0.00},  # Low socio-economic class
        '2': {'15-19': 0.003, '20-24': 0.026, '25-29': 0.074, '30-34': 0.069, '35-39': 0.023, '40-44': 0.004, '45-49': 0.00},  # Middle socio-economic class
        '3': {'15-19': 0.002, '20-24': 0.018, '25-29': 0.071, '30-34': 0.074, '35-39': 0.03, '40-44': 0.006, '45-49': 0.00}  # High socio-economic class
    }

    # Initialize the total children count for each agent
    agents_df['Children'] = agents_df['Children'] 
    
    # Iterate through each agent in the DataFrame
    for index, agent in agents_df.iterrows():
        # Increment agent's age by 1
        agents_df.at[index, 'Age'] += 1
        
        # Extract socio-economic class and age of the agent
        socio_economic_class = agent['socio_economic_class']
        age = agent['Age']
        
        # Determine the age bin of the agent
        if age <= 19:
            age_bin = '15-19'
        elif age <= 24:
            age_bin = '20-24'
        elif age <= 29:
            age_bin = '25-29'
        elif age <= 34:
            age_bin = '30-34'
        elif age <= 39:
            age_bin = '35-39'
        elif age <= 44:
            age_bin = '40-44'
        else:
            age_bin = '45-49'
        
        # Compute the probability of having a child based on socio-economic class and age
        probability = probability_weights.get(socio_economic_class, {}).get(age_bin, 0)
        
        # Simulate if the agent has a child based on the computed probability
        has_child = np.random.choice([True, False], p=[probability, 1 - probability])
        
        # If agent has a child, mark it in the corresponding age bin column
        if has_child:
            agents_df.at[index, age_bin] += 1
            agents_df.at[index, 'Children'] += 1
            
            
    return agents_df


# Example usage:
# agents_df = pd.DataFrame({...})
# agents_df = simulate_child_birth(agents_df, 1)


# Example usage:
# Assume agents_df is the DataFrame containing agent data
# agents_df = pd.DataFrame({...})


# Simulate for 10 years
for year in range(1, 30):
    # Simulate child birth for all agents for the current year
    agents_df_10k = simulate_child_birth(agents_df, year)
    
    # Save the data to a CSV file
    agents_df_10k.to_csv(f'agents_data.csv', index=False)


""""
# Calculate the amount of childen born in each age bin
children_by_age = {}
for i in range(0, len(agents_data_10k.columns), 5):
    columns = agents_data_10k.columns[i:i+5]
    sum_of_children = agents_data_10k[columns].sum().sum()
    age_group = f'Age Group {i//5 + 1}'
    children_by_age[age_group] = sum_of_children

"""