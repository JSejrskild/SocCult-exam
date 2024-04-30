import random
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


########## PREPERATION ##########
# defining agents as abstact class
class Agent:
    # init-method, the constructor method for agents
    # maybe we dont need the position and index parameters? these will be given in the populate_company function
    def __init__(self, position, index, gender, age, senority):
        self.position = position
        self.index = index
        self.gender = gender
        self.age = age
        self.senority = senority


# function for creating empty dictionary (company)
def create_company(titles:list, n:list):
    '''
    Creates a dictionary with the given titles as keys and empty lists as values.

    Parameters
    ----------
    titles : list, a list of strings with job titles
    n : list, a list of integers with the number of agents in each job title
    '''
    company = {}
    for i in range(len(titles)):
        key = titles[i] 
        value = [None for i in range(0, n[i])]
        company[key] = value

    return company



# function for creating the first agents and putting them in the company
def populate_company(company: dict):
    '''
    Creates the first agents and puts them in the company dictionary.

    Parameters
    ----------
    company : dictionary, a dictionary with the job titles as keys and empty lists as values
    '''
    for i in company.keys():
        for j in range(0, len(company[i])):
            if i == 'Department Head':
                weights = [0.8, 0.2] # more likely to be male when department head
                age = random.gauss(50, 8)
                senority = random.gauss(10, 3)
            elif i == 'Leader':
                weights = [0.7, 0.3] # more likely to be male when leader
                age = random.gauss(40, 6)
                senority = random.gauss(5, 3)
            elif i == 'Senior':
                weights = [0.6, 0.4] # more likely to be male when senior
                age = random.gauss(35, 6)
                senority = random.gauss(4, 1)
            elif i == 'Junior': 
                weights = [0.5, 0.5] # equally likely to be male and female
                age = random.gauss(30, 6)
                senority = random.gauss(3, 1)

            company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weights, k = 1), age = age, senority = senority)


# function for counting gender of agents in the different hierchical levels of the company
def count_gender(company):
    '''
    Counts the number of males and females of each job-title in the company

    Parameters
    ---------- 
    company : dictionary, a dictionary with the job titles as keys and empty lists as values
    '''
    female_count = np.repeat(0, len(company.keys()))
    male_count = np.repeat(0, len(company.keys()))

    for index, i in enumerate(company.keys()):
        for j in range(0, len(company[i])):
            if company[i][j] is not None:
                if company[i][j].gender == ['female']:
                    female_count[index] += 1
                else:
                    male_count[index] += 1
    return female_count, male_count


########## PLOTTING ##########
# function for plotting the gender distribution in the company
def plot_gender(company, tick):
    '''
    Plots the current gender distribution in each job-title of the company

    Parameters
    ---------- 
    company : dictionary, a dictionary with the job titles as keys and lists of agents as values
    '''
    female, male = count_gender(company)
    
    labels = company.keys()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(9,6))
    rects1 = ax.bar(x - width/2, female, width, label='Female', color = 'skyblue')
    rects2 = ax.bar(x + width/2, male, width, label='Male', color = 'steelblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Count by Job-title and Gender, month = {}'.format(tick + 1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    return ax

# function for plotting the gender distribution development over all ticks
def plot_gender_development(company: dict, months:int, data: pd.DataFrame):
    '''
    Plots the counts of each gender in each job-title in the company for each tick

    Parameters
    ----------
    company : dictionary, a dictionary with the job titles as keys and lists of agents as values
    months: int, the number of months simulated
    data: pandas dataframe, a dataframe with the information about the genders of employees in each job-title
    '''
    fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize=(12,5))
    colorlist = {'female': 'skyblue', 'male': 'steelblue'}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colorlist.values()]
    plt.legend(markers, colorlist.keys(), numpoints=1) # legend
    fig.suptitle('Counts of each gender in each position', fontsize=16) # title
    fig.text(0.5, 0.04, 'Month', ha='center', va='center') # x-axis label 
    fig.text(0.07, 0.5, 'Count', ha='center', va='center', rotation='vertical') # y-axis label

    for index, ax in enumerate(axs.flatten()):
        jobtitle = list(company.keys())[index]
        #ax.set_xlabel('Month')
        #ax.set_ylabel('Count')
        ax.set_title(jobtitle) # setting title to the job-title

        for gender in ['female', 'male']:
            gender_dat = data.loc[data['gender'] == gender]
            dat = gender_dat[jobtitle]
            ticks = gender_dat['tick']
            ax.plot(ticks, dat, color = colorlist[gender])
    





########## SIMULATION ##########

# NOTES
# check savepath in beginning of simulation
# figure out what data to include


def run_abm(months: int, save_path: str, company_titles: list, titles_n: list):
    '''
    Runs the ABM simulation

    Parameters
    ----------
    months : int, the number of months to simulate
    save_path : str, the path to the csv file
    company_titles : list, a list of strings with the job titles
    titles_n : list, a list of integers with the number of agents in each job title
    '''
    # creating empty dataframe for the results
    col_names = ['tick', 'gender']
    col_names.extend(company_titles)
    data = pd.DataFrame(columns = col_names)

    # create company
    company = create_company(company_titles, titles_n)
    # populate company
    populate_company(company)
    # plot initial 
    plot_gender(company, tick=-1)

    for month in range(months):
        # iterating through all agents
        for ind_i, i in enumerate(company.keys()):
            for j in range(0, len(company[i])):
                if company[i][j] is not None:
                    # adding a month to the senority of all agents
                    company[i][j].senority += 1/12
                    # adding a month to the age of all agents
                    company[i][j].age += 1/12
                    # if the agent has reached the age of 68, he should be fired
                    if company[i][j].age >= 68:
                        company[i][j] = None

                # if there is an empty position in the company, a agent should be promoted or created
                # REMEMBER: maybe we should retain information about the previous agent to hire someone similar to? eg. instead of setting the value in the dictionary to None we could set it to gender of the old agent? maybe also include age?
                    # but we would have to change all the places where it says if company.. == None
                if company[i][j] == None:
                    # if level is the lowest, a new agent is created
                    if i == list(company.keys())[-1]:
                        company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = [0.5, 0.5], k = 1), age = random.gauss(30, 6), senority = random.gauss(3, 1))

                    # choosing another agent from the lower level of the company to promote (e.g. junior to senior)
                    else:
                        # choosing a random agent from the lower level of the company 
                        # REMEMBER: SHOULD BE CHANGED TO CHOOSE THE BEST AGENT, possibly by age or senority or adding some kind of discrimination
                        agent = None
                        while agent == None: # REMEMBER: This runs infinetly if all agents are None in the given category... FIX THIS
                            agent = random.choice(company[list(company.keys())[ind_i+1]])


                        # promoting the agent
                        company[i][j] = agent
                        # removing the agent from the lower level of the company
                        company[agent.position][agent.index] = None
                        # changing the position of the agent
                        company[i][j].postition = i
                        # changing the index of the agent
                        company[i][j].index = j


            ### WE NEED TO FIND ANOTHER WAY FOR AGENTS TO QUIT OR GET FIRED ###    
            # fire one random department head if there are more than 18 department heads
            if i == 'Department Head' and len(company[i]) > 19:
                company[i][random.randint(0, len(company[i])-1)] = None

            # fire 1 random leader if there are more than 39 leaders
            if i == 'Leader' and len(company[i]) > 39:
                company[i][random.randint(0, len(company[i])-1)] = None
            
            #fire 1 random senior if there are more than 69 senior
            if i == 'Senior' and len(company[i]) > 69:
                company[i][random.randint(0, len(company[i])-1)] = None

            # fire 1 random junior if there are more than 128 juniors
            if i == 'Junior' and len(company[i]) > 128:
                company[i][random.randint(0, len(company[i])-1)] = None
            


        # plotting and appending data to data frame                           
        # plot_gender(company, tick = month)

        f = {'gender': 'female', 'tick': month, 'Department Head': count_gender(company)[0][0],'Leader': count_gender(company)[0][1], 'Senior': count_gender(company)[0][2], 'Junior': count_gender(company)[0][3]}
        m = {'gender': 'male', 'tick': month, 'Department Head': count_gender(company)[1][0],'Leader': count_gender(company)[1][1], 'Senior': count_gender(company)[1][2], 'Junior': count_gender(company)[1][3]}
        
        # create pandas dataframe from dictionaries f and m
        new_data = pd.DataFrame.from_dict([f, m])
        data = data.append(new_data, ignore_index=False, verify_integrity=False, sort=False)

        print('tick {} done'.format(month))

    # plotting the gender development over time
    plot_gender_development(company, months=months, data = data)
    # saving the data to a csv-file
    data.to_csv(save_path)


                