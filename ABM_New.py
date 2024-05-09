import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('Fertility_clean.csv')  # Replace with the actual filename and path

# Define Agent class
class Agent:
    def __init__(self, municipality, fertility_rate, socio_economic_class, age, children):
        self.municipality = municipality
        self.fertility_rate = fertility_rate
        self.socio_economic_class = socio_economic_class
        self.age = age
        self.children = 0

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

# Define a function to create Municipality classes dynamically
def create_municipality_class(name, fertility_rate, initial_population_percentage, socio_economic_class, socio_economic_weights):
    municipality_class = type(name, (object,), {
        'name': name,
        'fertility_rate': fertility_rate,
        'initial_population_percentage': initial_population_percentage,
        'socio_economic_class': socio_economic_class,
        'agents_data': []  # List to store agent data within the municipality
    })

    # Generate agent data within the municipality
    mean_fertility_rate = fertility_rate
    std_dev_fertility_rate = 0.1 * fertility_rate  # Adjust the standard deviation as needed
    total_agents = 1000  # Total number of agents per municipality
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

# Example usage - output municipalities
for municipality_name, municipality_class in municipality_classes.items():
    print(f"Municipality: {municipality_name}")
    print(f"Fertility Rate: {municipality_class.fertility_rate}")
    print(f"Socio-economic Class: {municipality_class.socio_economic_class}")
    print(f"Initial Population Percentage: {municipality_class.initial_population_percentage}")
    print(f"Number of Agents: {len(municipality_class.agents)}")
    print(f"Agents:")
    for agent in municipality_class.agents:
        print(f"  - Fertility Rate: {agent.fertility_rate}, Socio-economic Class: {agent.socio_economic_class}, Age: {agent.age}, Municipality: {agent.municipality.name}")
    print()

# STEP 05 - Create DF

# Assuming you have already created the municipality_classes dictionary

# Initialize an empty list to collect agent attributes
agent_data_list = []

# Iterate through municipality classes and their agents
for municipality_name, municipality_class in municipality_classes.items():
    for agent in municipality_class.agents:
        # Collect agent attributes into a dictionary
        agent_data = {
            'Municipality': agent.municipality.name,
            'Fertility Rate': agent.fertility_rate,
            'Socio': agent.socio_economic_class,
            'Age': agent.age,
            'Children': agent.children
        }
        # Append agent data to the list
        agent_data_list.append(agent_data)

# Create DataFrame from the list of agent data
agents_df = pd.DataFrame(agent_data_list)

# Display the DataFrame
print(agents_df)

agents_df.to_csv("agents_df1")


# SIMULATION


def simulate_child_birth(agents_df):
    # Define probability weights for having a child based on age bins and socio-economic classes
    probability_weights = {
        '1': {'15-25': 1, '26-35': 0.2, '36-45': 0.3},  # Low socio-economic class
        '2': {'15-25': 0.05, '26-35': 0.15, '36-45': 0.25},  # Middle socio-economic class
        '3': {'15-25': 0.5, '26-35': 0.1, '36-45': 0.2}  # High socio-economic class
    }
    
    # Iterate through each agent in the DataFrame
    for index, agent in agents_df.iterrows():
        # Extract socio-economic class and age of the agent
        socio_economic_class = agent['Socio']
        age = agent['Age']
        
        # Determine the age bin of the agent
        if age <= 25:
            age_bin = '15-25'
        elif age <= 35:
            age_bin = '26-35'
        else:
            age_bin = '36-45'
        
        # Compute the probability of having a child based on socio-economic class and age
        probability = probability_weights.get(socio_economic_class, {}).get(age_bin, 0)
        
        # Simulate if the agent has a child based on the computed probability
        has_child = np.random.choice([True, False], p=[probability, 1 - probability])
        
        # Increment the number of children for the agent if they had a child
        if has_child:
            agents_df.at[index, 'Children'] += 1

        # Increment agent's age by 1
        agents_df.at[index, 'Age'] += 1
            
    return agents_df

# Example usage:
# Assume agents_df is the DataFrame containing agent data
# agents_df = pd.DataFrame({...})

# Create DataFrame for agents
agents_df = pd.DataFrame(data)

# Simulate for 10 years
for year in range(10):
    # Simulate child birth for all agents
    agents_df = simulate_child_birth(agents_df)
    
    # Save the data to a CSV file
    agents_df.to_csv(f'agents_data_year_{year + 1}.csv', index=False)
    
    # Display message indicating the completion of the simulation for the current year
    print(f"Simulation for year {year + 1} completed. Data saved to agents_data_year_{year + 1}.csv")


print(agents_df.columns)
