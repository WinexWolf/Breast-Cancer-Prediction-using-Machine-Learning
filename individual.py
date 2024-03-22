from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

class Individual:
    def __init__(self, num_features):
        self.chromosome = ''.join([str(random.randint(0, 1)) for _ in range(num_features)])
        self.fitness = None

    def calculate_fitness(self, X, y):
        # Convert binary chromosome to boolean mask
        feature_mask = [bool(int(bit)) for bit in self.chromosome]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X[:, feature_mask], y, test_size=0.2)

        # Train a decision tree classifier on the selected features
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        # Calculate fitness as accuracy of the classifier
        y_pred = clf.predict(X_test)
        self.fitness = accuracy_score(y_test, y_pred)

    def crossover(self, partner):
        # Create a new child
        child = Individual(len(self.genes))
        
        # Choose a random midpoint
        midpoint = random.randint(0, len(self.genes))
        
        # Combine the genes of the two parents to create the child's genes
        child.genes = self.genes[:midpoint] + partner.genes[midpoint:]
        
        return child
