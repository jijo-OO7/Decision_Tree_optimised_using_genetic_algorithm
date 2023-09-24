import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# Load your user-defined dataset
# Replace this with your dataset loading code
# X, y = load_your_dataset()

# For this example, we'll use the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fitness function
def evaluate(individual):
    # Create a Decision Tree Classifier with user-defined parameters
    max_depth, min_samples_split, criterion = individual
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy,

# Define the Genetic Algorithm parameters
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Define parameter ranges for optimization
toolbox.register("attr_float", np.random.uniform, 1, 32)  # Max Depth
toolbox.register("attr_int", np.random.randint, 2, 11)   # Min Samples Split
toolbox.register("attr_choice", np.random.choice, ["gini", "entropy"])  # Criterion

# Create individuals with user-defined parameter ranges
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float, toolbox.attr_int, toolbox.attr_choice), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population_size = 10
    generations = 5
    cxpb = 0.7
    mutpb = 0.2

    population = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size,
                             cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=None, halloffame=None, verbose=True)

    best_individual = tools.selBest(population, k=1)[0]
    best_max_depth, best_min_samples_split, best_criterion = best_individual
    best_classifier = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split, criterion=best_criterion)
    best_classifier.fit(X_train, y_train)
    y_pred = best_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best Decision Tree Parameters: Max Depth={best_max_depth}, Min Samples Split={best_min_samples_split}, Criterion={best_criterion}")
    print(f"Accuracy on Test Data: {accuracy:.2f}")

if __name__ == "__main__":
    main()
