import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('fertility.csv')  # Replace with the actual filename and path

data['Samlet_fertilitet'] = data['Samlet_fertilitet'].astype(int)

# Define Agent class
class Agent:
    def __init__(self, municipality, fertility_rate, socio_economic_class, age, children):
        self.municipality = municipality
        self.fertility_rate = fertility_rate
        self.socio_economic_class = socio_economic_class
        self.age = age
        self.children = 0

# Define a function to create agents for a municipality
def create_agents_for_municipality(municipality_class, municipality):
    agents = []
    for agent_data in municipality_class.agents_data:
        fertility_rate = agent_data['fertility_rate']
        socio_economic_class = agent_data['socio_economic_class']
        age = agent_data['age']
        agents.append(Agent(
            municipality=municipality,
            fertility_rate=fertility_rate,
            socio_economic_class=socio_economic_class,
            age=age,
            children=0
        ))
    return agents

# Define a function to create Municipality classes dynamically
def create_municipality_class(name, fertility_rate, initial_population, socio_economic_class, socio_economic_weights):
    municipality_class = type(name, (object,), {
        'name': name,
        'fertility_rate': fertility_rate,
        'initial_population': initial_population,
        'socio_economic_class': socio_economic_class,
        'agents_data': []  # List to store agent data within the municipality
    })

    # Generate agent data within the municipality
    mean_fertility_rate = fertility_rate
    std_dev_fertility_rate = 0.1 * fertility_rate  # Adjust the standard deviation as needed
    for _ in range(initial_population):
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
    municipality_class.agents = create_agents_for_municipality(municipality_class, municipality_class)
    
    return municipality_class

# Create Municipality classes dynamically based on the dataset
municipality_classes = {}
for index, row in data.iterrows():
    municipality_name = row['municipalities']
    fertility_rate = row['fertilitetsrate']
    initial_population = row['Samlet_fertilitet']  # Assuming 'Population' is the column name for population size
    socio_economic_class = row['econ_score']  # Assuming 'Socio-economic Class' is the column name
    
    # Define weights for socio-economic classes for this municipality
    if socio_economic_class == '1':
        socio_economic_weights = [0.6, 0.2, 0.2]  # Low socio-economic class has a weight of 0.6
    elif socio_economic_class == '2':
        socio_economic_weights = [0.2, 0.6, 0.2]  # Middle socio-economic class has a weight of 0.6
    else:
        socio_economic_weights = [0.2, 0.2, 0.6]  # High socio-economic class has a weight of 0.6
        
    municipality_class = create_municipality_class(municipality_name, fertility_rate, initial_population, socio_economic_class, socio_economic_weights)
    municipality_classes[municipality_name] = municipality_class

# Example usage - output municipalities
for municipality_name, municipality_class in municipality_classes.items():
    print(f"Municipality: {municipality_name}")
    print(f"Fertility Rate: {municipality_class.fertility_rate}")
    print(f"Socio-economic Class: {municipality_class.socio_economic_class}")
    print(f"Initial Population: {municipality_class.initial_population}")
    print(f"Number of Agents: {len(municipality_class.agents)}")
    print(f"Agents:")
    for agent in municipality_class.agents:
        print(f"  - Fertility Rate: {agent.fertility_rate}, Socio-economic Class: {agent.socio_economic_class}, Age: {agent.age}, Municipality: {agent.municipality.name}")
    print()
