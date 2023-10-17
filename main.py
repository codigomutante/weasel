import random
from deap import base, creator, tools
import time

ELITE_SIZE = 1
TARGET = "METHINKS IT IS LIKE A WEASEL"
POP_SIZE = 100
MUTPB = 0.05
NGEN = 1000

# Definición del problema como problema de maximización
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_char", lambda: random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ "))
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_char, len(TARGET)
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def mutateCharacters(individual, mutation_prob=MUTPB):
    """
    Mutar varios caracteres del individuo en función de una probabilidad de mutación determinada.
    """
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    return (individual,)


def fitness(mutant, target):
    """
    Obtiene el score que una cadena mutante tiene con respecto al objetivo.
    """
    score = 0
    length = len(target)

    # Comprobamos la igualdad de caracteres para cada posición en las cadenas
    for i in range(length):
        if target[i] == mutant[i]:
            score += 1

    # Devolvemos el score como una fracción del total de caracteres en target
    return score / length


def evalSimilarity(individual):
    return (fitness("".join(individual), TARGET),)


toolbox.register("evaluate", evalSimilarity)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutateCharacters)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    # Temporizador
    start_time = time.time()

    # Initialize the population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluar la población
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(NGEN):
        # Seleccionar ELITE_SIZE mejores individuos y guardarlos
        elites = tools.selBest(pop, ELITE_SIZE)

        offspring = []
        for _ in range(POP_SIZE - ELITE_SIZE):
            child = toolbox.clone(elites[0])
            toolbox.mutate(child)
            del child.fitness.values
            offspring.append(child)

        # Añadir los elites al offspring
        offspring += elites

        # Evaluar individuos sin valor de fitness
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Mostrar el mejor resultado en la generación actual
        fits = [ind.fitness.values[0] for ind in pop]
        print("Generación {}: Max: {}".format(gen, max(fits)))
        current_best_ind = tools.selBest(pop, 1)[0]
        print("Mejor individuo: {}".format("".join(current_best_ind)))
        # Si encontramos un individuo perfecto, detenemos la evolución

        if max(fits) == 1.0:
            print("Encontrada frase objetivo: {}".format(TARGET))
            print("Generación: {}".format(gen))
            print("Tiempo de ejecución: %s seconds" % (time.time() - start_time))
            exit(0)

    print("No se ha encontrado la frase")
    best_ind = tools.selBest(pop, 1)[0]
    print(
        "La mejor es : {}\nCon un fitness de: {}".format(
            "".join(best_ind), best_ind.fitness.values
        )
    )
    print("Generación: {}".format(gen))
    print("Tiempo de ejecución: %s seconds" % (time.time() - start_time))
    exit(0)


if __name__ == "__main__":
    main()
