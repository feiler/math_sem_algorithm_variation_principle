import random

def order_crossover(parent1, parent2):
    # Stellen Sie sicher, dass die Eltern dieselbe Länge haben
    assert len(parent1) == len(parent2)
    
    size = len(parent1)
    
    # Wählen Sie zwei zufällige Punkte für die Kreuzung
    start, end = sorted(random.sample(range(size), 2))
    
    # Kind 1 von Parent 1 und Parent 2
    child1 = [None]*size
    child1[start:end] = parent1[start:end]
    
    pointer = end
    for city in parent2[end:] + parent2[:end]:
        if city not in child1:
            if pointer == size:
                pointer = 0
            child1[pointer] = city
            pointer += 1

    # Kind 2 von Parent 2 und Parent 1
    child2 = [None]*size
    child2[start:end] = parent2[start:end]
    
    pointer = end
    for city in parent1[end:] + parent1[:end]:
        if city not in child2:
            if pointer == size:
                pointer = 0
            child2[pointer] = city
            pointer += 1
    
    return child1, child2

def pmx_crossover(parent1, parent2):
    # Stellen Sie sicher, dass die Eltern dieselbe Länge haben
    assert len(parent1) == len(parent2)
    
    size = len(parent1)
    
    # Wählen Sie zwei zufällige Punkte für die Kreuzung
    start, end = sorted(random.sample(range(size), 2))
    
    def pmx(parent1, parent2):
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        for i in range(start, end):
            if parent2[i] not in child:
                position = i
                while start <= position < end:
                    position = parent2.index(parent1[position])
                child[position] = parent2[i]
        
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]
        
        return child
    
    child1 = pmx(parent1, parent2)
    child2 = pmx(parent2, parent1)
    
    return child1, child2

def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [None]*size, [None]*size

    def make_cycle(parent1, parent2, child):
        start = 0
        while None in child:
            if child[start] is None:
                indices = []
                val = parent1[start]
                while True:
                    indices.append(start)
                    start = parent1.index(parent2[start])
                    if start in indices:
                        break
                for i in indices:
                    child[i] = parent1[i]

    make_cycle(parent1, parent2, child1)
    make_cycle(parent2, parent1, child2)

    return child1, child2

# Beispiel-Eltern (Reihenfolge der Städte)
parent1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
parent2 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# Order Crossover
child1, child2 = order_crossover(parent1, parent2)
print("Order Crossover:")
print("Child 1:", child1)
print("Child 2:", child2)

# Partially Mapped Crossover
child1, child2 = pmx_crossover(parent1, parent2)
print("Partially Mapped Crossover:")
print("Child 1:", child1)
print("Child 2:", child2)

# Cycle Crossover
child1, child2 = cycle_crossover(parent1, parent2)
print("Cycle Crossover:")
print("Child 1:", child1)
print("Child 2:", child2)
